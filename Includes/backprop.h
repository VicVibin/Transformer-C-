#pragma once
#include <unordered_set>
#include <future>
#include <string>
#include "MathGPU.h"

using str = std::string;
struct Node;
using graph = std::shared_ptr<Node>;
using graph_tree = std::vector<graph>;
struct AdamParameter;
class GraphOperations;

graph_tree topological_sort(const graph& root);

// Node definition
struct Node {
    str op_name;
    graph_tree inputs;
    Matrix_d output;
    Matrix_d grad;
    std::function<void()> backward;

    Node(str name);
};

struct AttentionHead
{   Linear_Algebra lin;
    std::shared_ptr<AdamParameter> W_q;
    std::shared_ptr<AdamParameter> W_k;
    std::shared_ptr<AdamParameter> W_v;
    int head_id;
    AttentionHead(int id, int d_model, int d_k);
    graph compute(graph input, GraphOperations& go, int stream_id);
    graph compute_d(graph Queries, graph KeysnVals, GraphOperations &go, int head_id);
};

// Parameter node
struct Parameter : public Node {
    float learning_rate;
    Parameter(str name, const Matrix_d& value, float lr = 0.001f);
    virtual void update();
};

// Adam parameter node
struct AdamParameter : public Parameter {
    Math math;
    Linear_Algebra lin;
    Matrix_d m, v;
    float beta1, beta2, epsilon;
    int t;

    AdamParameter(str name, const Matrix_d& value, float lr = 0.001f);
    void update() override;
};

class GraphOperations {
private:
    static Linear_Algebra lin;
    static Machine_Learning ml;
    static Math math;

public:
    static Linear_Algebra_GPU linG;
    static graph concatenate(const std::vector<graph> & inputs);
    static graph scale(graph inputs, float scaling);
    static graph transpose(const graph &input);
    static graph add(const graph& A, const graph& B);
    static graph matmul(const graph& A, const graph& B, int stream_id = 0);
    static graph RelU(const graph& input);
    static graph Softmax(const graph& input);
    static graph Softmask(const graph & A);
    static graph LayerNorm(const graph & A);
    void backprop(graph loss_node);
    static graph MSE(const graph & pred, const Matrix_d& output);
    static graph SoftmaxCrossEntropy(const graph & pred, const Matrix_d& output);
    void clear_graph(graph loss_node);
};

class MultiHeadAttention
{   Linear_Algebra lin;
    Math math;
    std::vector<AttentionHead> heads;
    std::shared_ptr<AdamParameter> W_o;
    int d_model;

public:
    Linear_Algebra_GPU LinG;
    MultiHeadAttention(int num_heads, int d_model);
    graph forward(graph input, GraphOperations& go);
    graph forward_d(graph queries_B, graph keys_values_A, GraphOperations& go);
    std::vector<std::shared_ptr<AdamParameter>> parameters();
};