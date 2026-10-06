#include <torch/extension.h>
#include "ggml.h"
#include "gguf.h"
#include "ggml-quants.h"
#include <omp.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdio>
#include <mutex>
#include <algorithm>

static std::unordered_map<std::string, std::vector<float>> g_imatrix_cache;
static std::string g_loaded_imatrix_file = "";
static std::mutex g_imatrix_mutex;

void load_gguf_imatrix(const std::string& fname) {
    std::lock_guard<std::mutex> lock(g_imatrix_mutex);
    if (g_loaded_imatrix_file == fname) return;
    g_loaded_imatrix_file = fname;
    g_imatrix_cache.clear();

    struct ggml_context * ctx = nullptr;
    struct gguf_init_params params = { false, &ctx };
    struct gguf_context * gctx = gguf_init_from_file(fname.c_str(), params);
    if (!gctx) throw std::runtime_error("Failed to load imatrix GGUF file: " + fname);

    int n_tensors = gguf_get_n_tensors(gctx);
    std::unordered_map<std::string, ggml_tensor*> tensors;
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gctx, i);
        tensors[name] = ggml_get_tensor(ctx, name);
    }

    for (auto & kv : tensors) {
        const std::string & name = kv.first;
        if (name.size() > 8 && name.compare(name.size() - 8, 8, ".in_sum2") == 0) {
            std::string base = name.substr(0, name.size() - 8);
            std::string counts_name = base + ".counts";
            if (tensors.count(counts_name)) {
                ggml_tensor * sum2_t = kv.second;
                ggml_tensor * counts_t = tensors[counts_name];
                if (sum2_t->type != GGML_TYPE_F32 || counts_t->type != GGML_TYPE_F32) continue;

                float * sum2 = (float*)sum2_t->data;
                float * counts = (float*)counts_t->data;
                int ncounts = ggml_nelements(counts_t);
                int nval = ggml_nelements(sum2_t);
                if (ncounts == 0) continue;
                int ne0 = nval / ncounts;

                std::vector<float> rms(nval);
                for (int j = 0; j < ncounts; ++j) {
                    float count = counts[j];
                    if (count > 0.0f) {
                        for (int k = 0; k < ne0; ++k) rms[j * ne0 + k] = sum2[j * ne0 + k] / count;
                    } else {
                        for (int k = 0; k < ne0; ++k) rms[j * ne0 + k] = 1.0f;
                    }
                }
                g_imatrix_cache[base] = std::move(rms);
            }
        }
    }
    gguf_free(gctx);
    ggml_free(ctx);
}

// --- NEW FUNCTION: Check if imatrix exists for a tensor ---
bool has_imatrix(const std::string& gguf_name, const std::string& imatrix_file) {
    if (!imatrix_file.empty()) load_gguf_imatrix(imatrix_file);
    std::lock_guard<std::mutex> lock(g_imatrix_mutex);
    if (gguf_name.empty()) return false;
    return g_imatrix_cache.count(gguf_name) > 0;
}
// ---------------------------------------------------------

torch::Tensor apply_llama_quant_noise(torch::Tensor weight, std::string gguf_name, int64_t ggml_type_id, std::string imatrix_file) {
    if (!imatrix_file.empty()) load_gguf_imatrix(imatrix_file);
    auto w_2d = weight.view({-1, weight.size(-1)}).contiguous().to(torch::kFloat32);
    int64_t nrows = w_2d.size(0);
    int64_t n_per_row = w_2d.size(1);
    
    size_t row_size = ggml_row_size((ggml_type)ggml_type_id, n_per_row);
    std::vector<uint8_t> q_buf(row_size * nrows);
    
    const float* imat = nullptr;
    int imat_size = 0;
    {
        std::lock_guard<std::mutex> lock(g_imatrix_mutex);
        if (!gguf_name.empty() && g_imatrix_cache.count(gguf_name)) {
            imat_size = g_imatrix_cache[gguf_name].size();
            if (imat_size >= n_per_row) imat = g_imatrix_cache[gguf_name].data();
        }
    }
    
    const float* src_ptr = w_2d.data_ptr<float>();
    uint8_t* dst_ptr = q_buf.data();
    
    int nmat = (imat && imat_size > n_per_row) ? (imat_size / n_per_row) : 1;
    if (nmat > 1 && nmat > nrows) nmat = nrows;
    int rows_per_mat = (nmat > 1) ? (nrows / nmat) : nrows;
    
    int64_t chunk_size = 32;
    if (nrows < chunk_size) chunk_size = nrows;
    if (chunk_size == 0) chunk_size = 1;
    
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < nrows; i += chunk_size) {
        int64_t current_rows = std::min(chunk_size, nrows - i);
        int m = (nmat > 1) ? (i / rows_per_mat) : 0;
        if (m >= nmat) m = nmat - 1;
        const float* current_imat = imat ? imat + m * n_per_row : nullptr;
        ggml_quantize_chunk((ggml_type)ggml_type_id, src_ptr, dst_ptr, i * n_per_row, current_rows, n_per_row, current_imat);
    }
    
    auto out_tensor = torch::empty_like(w_2d);
    float* dst = out_tensor.data_ptr<float>();
    const ggml_type_traits* traits = ggml_get_type_traits((ggml_type)ggml_type_id);
    if (traits->to_float) {
#pragma omp parallel for schedule(dynamic, 32)
        for (int64_t i = 0; i < nrows; ++i) {
            traits->to_float(q_buf.data() + i * row_size, dst + i * n_per_row, n_per_row);
        }
    } else {
        throw std::runtime_error("Unsupported ggml type for dequantization");
    }
    return out_tensor.view_as(weight).to(weight.scalar_type());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_llama_quant_noise", &apply_llama_quant_noise, "Apply native llama.cpp quantization noise");
    m.def("has_imatrix", &has_imatrix, "Check if imatrix data exists for a specific tensor"); // Exposed to Python
}
