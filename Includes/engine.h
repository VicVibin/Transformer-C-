#pragma once
#include "debugging_utils.h"
#include "kernels.h"

/*
#include <cudnn.h> 
*/

#include <functional>
#include <memory>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <opencv2/opencv.hpp>

struct NodeBackProp;
using graph = std::shared_ptr<NodeBackProp>;
using graph_tree = std::vector<graph>;
using BatchText = std::vector<Text>;

void isNan(const str name, const float* X, const long long total);

struct NodeBackProp 
{
    str op_name;
    graph_tree inputs;
    float*  output;
    float*  grad;
    int dim[4];
    long long total;
    bool owns_output;
    std::function<void()> forward;
    std::function<void()> backward;
    std::function<void()> free;
    std::function<void()> zero_grad;
    std::function<void()> updateParams;
    std::function<void(const double*)> clipnorm;
    std::function<void(double*)> accumulate;
    std::function<void()> printparams;
    virtual void update(const bool W=ADAMW){};
    
    NodeBackProp(str name, int batch, int d1, int d2, int d3, int allocation);

    void clear();
    void reshape(std::vector<int> new_dims);

};

struct Ipointer
{
    float *memory;
    int dimensions[4];
    long long total_size;
    Text labels;
};

Ipointer i2p(const std::string& filepath, int row_size = 0, int col_size = 0);
cv::Mat p2i(Ipointer X);
graph i2n(std::string filepath, int row_size = 0, int col_size = 0);
cv::Mat n2i(const graph X);
Ipointer Bi2p(const str& folder, const int num_images, const int row_size, const int col_size);
graph Bi2n(str filepath,const int num_images, const int row_size , const int col_size);
cv::Mat Bn2i(const graph X);

struct AdamParameter : public NodeBackProp {
    float *m, *v; int t; 
    float lr, b1, b2, epsilon; 
    long long total_size;
    int batch_size;
    float group_norm;
    float weight_decay;
    
    AdamParameter(str n, int batch, int out, int in, int row, int col, double norm = NORM);
    
    void save(std::ofstream& f) const;

    void load(std::ifstream& f); // Assumes the class structure is the exact same as when saved, otherwise may cause issues
    
    void update(const bool W = ADAMW) override;

    void accumulate_grad(double* global_scale);

    void gradnorm(const double* global_scale);

};

void Dimension(graph X);
void Dimension(AdamParameter* X);
void Zerograd(AdamParameter* X);
void Zerograd(graph X);
void isNan(graph X, const int type = 0);
void isNan(AdamParameter* X, const int type = 0);
void printGPU(const graph X, const int type = 0);
void printGPU(AdamParameter* X, const int type = 0);
void printHeadGPU(const graph X, const int type = 0);
void printHeadGPU(AdamParameter* X, const int type = 0);
int ArgMaxToCPU(const graph& input, int* X);
int TopKSampleToCPU(const graph& input, int* X, const int k);
void BMinMaxNorm(const graph &X);
void BMaxNorm(const graph & X);
void StandardNorm(const graph &X, const float img_max=255.0f, const float mean=0.5f, const float std=0.5f);
void StandardDeNorm(const graph &X, const float img_max=255.0f, const float mean=0.5f, const float std=0.5f);
void BPrintImage(const graph &X, const int row_size =0, const int col_size = 0);
void prepare(const graph &base, const graph &input, const graph &target, int t, int T, const uint64_t seed);
graph_tree topological_sort(const graph& root);

class TextualEmbedding
{
public:
    /*
    Parameters:
    1.) embed_dim: dimension of each word embedding
    2.) batch_size: maximum batch size for input text
    3.) max_c_len: maximum context length (number of words in each input text)
    4.) max_vocab_size: maximum vocabulary size (number of unique words in the vocabulary
    */
    float* EmbedSpace;      // GPU embedding memory space [MAX_VOCAB_SIZE x embed_dim]
    int* keys;              // GPU word index mapping to the embedding: [MAX_BATCH_SIZE x MAX_CONTEXT_LEN], changes each epoch
    Text input_text;        // Current input text batch
    std::unordered_map<str, int> WordSpace; // CPU word to key mapping
    std::unordered_map<int, str> KeySpace; // CPU key to word mapping
    std::unordered_set<str> Vocabulary; // Total set of all unique words which the index is the key
    const int MAX_BATCH_SIZE, MAX_CONTEXT_LEN, MAX_VOCAB_SIZE;
    const int embed_dim;
    const int tpb = THREADSPERBLOCK; 
    
private:
    str preprocessWord(const str& word);
public:
    TextualEmbedding(const int embed_dim, const int batch_size=128, const int max_c_len=64, const int max_vocab_size=10000);
    ~TextualEmbedding();
    void updateVocabulary(const Text& texts);
    void encodeText(const Text& texts, const int batch_idx = 0);
    void rencodeText(const Text& texts, const int batch_idx = 0, const int start_idx = 0);
    void encodeBatch(const BatchText& batch_texts);
    void forward(const graph& X);
    void rforward(const graph&X, const int start_idx = 0);
    void one_hot_forward(const graph&X);
    void EmbeddingUpdate(const graph&X);
};

class GraphOperations{
public:
    graph_tree nodes;
    double GB = 0;
    float loss;
    graph like(const graph& X, const str name = "");
    graph Last(const graph& X);
    graph PositionalEncoding(const int &t, const int d_model);
    graph MatrixPositionalEncoding(const graph& X, const int start_idx = 0);
    graph Broadcast_Add(const graph& A, const graph& B);
    graph Broadcast_Channel(const graph& A, const graph& B);
    graph Add(const graph& A, const graph& B);
    graph MeanSquaredError(const graph& prediction, const graph& target);
    graph CrossEntropy(const graph& prediction, const graph& target);
    graph SoftMaxCrossEntropy(const graph& prediction, const graph& target);
    graph Permute(const graph& X, int i0, int i1, int i2, int i3);
    graph BMM(const graph& A, const graph& B); // m x n, n x p = m x p
    graph BMMABT(const graph& A, const graph& B); //  m x n, p x n = m x p
    graph BMMATB(const graph& A, const graph& B); // m x n, m x p = n x p
    graph BMMATBT(const graph& A, const graph& B); // m x n, p x m = n x p
    graph SOFTMAX(const graph& X, const int type = 0); // type 0: row-wise, type 1: column-wise
    graph SOFTMASK(const graph& X, const int type = 0); // type 0: row-wise, type 1: column-wise
    graph Scale(const graph& input, const float scale);
    graph RELU(const graph& input);
    graph SILU(const graph& input);
    graph LeakyRELU(const graph& input);
    graph CopyCrop(const graph& input1, const graph& input2);
    graph CopyConcat(const graph& input1, const graph& input2);
    graph LayerNorm(const graph& X);
    graph BatchNorm(const graph& X);
    graph GroupNorm(const graph& X, const int group=8);
    graph InstanceNorm(const graph & X);
    void clipNorm(double* global_scale);
    void accumulate(double* global_scale); 
    void ParameterUpdate();
    void forward();
    void backward();
    void zero_grad();
    void printNodes(const bool display_grad=false);
    void clear_graph();
};

class Identity{   
private:
    GraphOperations &go;
    str name;
public: 
    Identity(GraphOperations& go_ref, const str name = "");
    graph forward(const graph& X);
};

class Linear
{ 
private: 
    int in, out;
public:
    GraphOperations &go;
    AdamParameter *W1;
    AdamParameter *B1;
    str op_name = "Linear Layer";
    Linear(GraphOperations &go_ref, const int input, const int output, const str name = "");
    graph forward(const graph & X);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
};

/*
struct ConvCache {
    int n = -1, c = -1, h = -1, w = -1;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    size_t workspace_size = 0;
};
*/

/*
class Conv2D {
private:
    GraphOperations& go;
    int out, inp, c, d, pad, stride;
    str name;
    AdamParameter *weights, *bias;


     * @brief For standard convolution call C2D (go, inp, out);
     * @param name go: GraphOperations reference
     * @param Input: number of input channels
     * @param Output: number of output channels
     * @param C: kernel size row: (default 3)
     * @param D: kernel size col: (default 3)
     * @param stride: stride size (default 1)
     * @param padding: padding size (default 1)
     * @param param: Name of the operation (default "" )

    // cuDNN Resources
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t x_desc, y_desc, bias_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    // Caching and Workspace
    ConvCache cache;
    float* d_workspace = nullptr;
    void ValidateCache(int n, int c_in, int h, int w);

public:
    Conv2D(GraphOperations& go_ref, int Input, int Output, int C, int D, int stride, int padding, str name);
    ~Conv2D();
    graph forward(const graph& X);
};

*/


class Convolute2D {
private:
    GraphOperations go;
    graph T_node;
    int inp, out, c, d, pad, stride;
public:
    AdamParameter *weights, *bias;
    str name;
    /**
     * @brief For standard convolution call C2D (go, inp, out);
     * @param name go: GraphOperations reference
     * @param Input: number of input channels
     * @param Output: number of output channels
     * @param C: kernel size row: (default 3)
     * @param D: kernel size col: (default 3)
     * @param stride: stride size (default 1)
     * @param padding: padding size (default 1)
     * @param param: Name of the operation (default "" )
     */
    Convolute2D(GraphOperations&go_ref, int Input, int Output, int C=3, int D=3, int stride = 1, int padding = 1, str param = "");
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X);
};

class Convolute2DT {
private:
    GraphOperations go;
    graph T_node;
    int inp, out, c, d, pad, stride;
public:
    AdamParameter *weights, *bias;
    str name;
    /**
     * @brief For transposed convolution call C2DT (go, inp, out);
     * @param name go: GraphOperations reference
     * @param Input: number of input channels
     * @param Output: number of output channels
     * @param C: kernel size row: (default 3)
     * @param D: kernel size col: (default 3)
     * @param stride: stride size (default 1)
     * @param padding: padding size (default 1)
     * @param param: Name of the operation (default "" )
     */
    Convolute2DT(GraphOperations&go_ref, int Input, int Output, int C=2, int D=2, int stride=2, int padding=0, str param="");
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X);
};

class VisionAttention 
{
public:
    GraphOperations &go; 
    GraphOperations subGraph;
    int batch, channels;
    Convolute2D *Q, *K, *V, *P;
    VisionAttention(GraphOperations &go_ref, const int Channels);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X_in);

};

class VisionCrossAttention
{
public:
    GraphOperations &go; 
    GraphOperations subGraph;
    int batch, channels, context_len, embed_dim;
    Convolute2D *Q, *K, *V, *P;
    VisionCrossAttention(GraphOperations &go_ref, const int Channels, const int ContextLen, const int EmbedDim);
    void save(std::ofstream& f) const;
    void load(std::ifstream& f);
    graph forward(const graph& X_in, const graph& Context);

};

void diffuse(float* input, float* model, float* theta, const long long total, const int t, const int T, const double s, const uint64_t seed);

void Noise(const graph & input);