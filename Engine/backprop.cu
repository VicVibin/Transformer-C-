#include "includes/backprop.h"
#include <tuple>

Linear_Algebra_GPU GraphOperations::linG;
Linear_Algebra GraphOperations::lin;
Machine_Learning GraphOperations::ml;
Math GraphOperations::math;

Node::Node(str name) : op_name(std::move(name)) {}

Parameter::Parameter(str name, const Matrix_d& value, float lr) : Node(name), learning_rate(lr) 
{
    output = value;
    grad = Matrix_d(value.size(), vector_d(value[0].size(), 0.0f));
}

void Parameter::update() 
{
    for (size_t i = 0; i < output.size(); ++i)
        for (size_t j = 0; j < output[0].size(); ++j)
            output[i][j] -= learning_rate * grad[i][j];

}

AdamParameter::AdamParameter(str name, const Matrix_d& value, float lr)
    : Parameter(name, value, lr), beta1(0.9f), beta2(0.999f), epsilon(1e-12f), t(0) 
{
    m = Matrix_d(value.size(), vector_d(value[0].size(), 0.0f));
    v = Matrix_d(value.size(), vector_d(value[0].size(), 0.0f));
}

void AdamParameter::update() 
{   
    ++t;
    for (size_t i = 0; i < output.size(); ++i) {
        for (size_t j = 0; j < output[0].size(); ++j) 
        {   m[i][j] = beta1 * m[i][j] + (1 - beta1) * grad[i][j];
            v[i][j] = beta2 * v[i][j] + (1 - beta2) * grad[i][j] * grad[i][j];
            const float m_hat = m[i][j] / (1 - math.power(beta1, t));
            const float v_hat = v[i][j] / (1 - math.power(beta2, t));
            output[i][j] -= learning_rate * m_hat / (math.sqrt(v_hat) + epsilon);
        }
    }
    if (output[0][0] != output[0][0])
    {   
        std::cout << "Value of m[0][0]: " << m[0][0] << "\n";
        std::cout << "Value of v[0][0]: " << v[0][0] << "\n";
        std::cout << "Value of grad[0][0]: " << grad[0][0] << "\n";
        throw std::invalid_argument("Output index is nan \n");
    }
}

// Topological sort function declaration
graph_tree topological_sort(const graph& root) 
{
    std::unordered_set<Node*> visited;
    graph_tree result;
    
    std::function<void(const graph&)> dfs = [&](const graph& node) 
    {
        if (!node || visited.count(node.get())) return;
        
        visited.insert(node.get());
        for (const auto& input : node->inputs) 
        {
            dfs(input);
        }
        result.push_back(node);
    };
    
    dfs(root);

 
    return result;
}

graph GraphOperations::add(const graph& A, const graph& B)
{
        auto node = std::make_shared<Node>("intermediate");
        node->inputs = {A, B};
        // Element-wise addition
        node->output = lin.force_add(A->output, B->output);
        node->backward = [node, A, B]() 
        {
            A->grad = node->grad;
            int row = B->output.size();
            int col = B->output[0].size();
            if(row == 1)
            {
                Matrix_d grad_sum(1, vector_d(col, 0.0f));
                for(const auto & row : node->grad)
                {
                    for(int j = 0; j < row.size(); j++)
                    {
                        grad_sum[0][j] += row[j]; 
                    }
                }
                B->grad = grad_sum;   
            }

            else
            {
            B->grad = node->grad;
            }
            
        };
        
        return node;
    }
  

    graph GraphOperations::LayerNorm(const graph &A)
    {
    auto node = std::make_shared<Node>("intermediate");
    node->inputs = {A};
    node->output = lin.LayerNorm(A->output);

    node->backward = [node, A]()
    {   
        Matrix_d grad_out = node->grad;
        Matrix_d  X = A->output;
        int n_rows = grad_out.size();
        int d = grad_out[0].size();  // assuming all rows have the same length

        A->grad.resize(n_rows, vector_d(d, 0.0f));

        for (int i = 0; i < n_rows; i++)
        {
            const vector_d &x = X[i];
            const vector_d &dy = grad_out[i];

            float mu = lin.mean(x);
            float std_inv = 1.0f / lin.std(x, 1e-5f);  // std_inv = 1 / sqrt(var + epsilon)

            vector_d x_hat(d), dx(d);
            float dot1 = 0.0f, dot2 = 0.0f;

            for (int j = 0; j < d; j++)
            {
                x_hat[j] = (x[j] - mu) * std_inv;
                dot1 += dy[j];
                dot2 += dy[j] * x_hat[j];
            }

            for (int j = 0; j < d; j++)
            {
                dx[j] = (1.0f / d) * std_inv *
                        (d * dy[j] - dot1 - x_hat[j] * dot2);
            }

            A->grad[i] = dx;
        }
    };

    return node;
}

graph GraphOperations::transpose(const graph & A)
{
    auto node = std::make_shared<Node>("intermediate");
    node->inputs = {A};
    node->output = lin.Transpose(A->output);
    node->backward = [node, A]()
    {   
        A->grad = lin.Transpose(node->grad);
    };
    return node;
}

graph GraphOperations::matmul(const graph& A, const graph& B, int stream_id)
    {           
        if (stream_id >= linG.stream_count()) 
        {
            stream_id = stream_id % linG.stream_count(); // Fall back to default stream
            // Or: stream_id = stream_id % linG.stream_count();
        }
        auto node = std::make_shared<Node>("intermediate");
        node->inputs = {A, B};
        node->output = linG.MatMul(A->output, B->output, stream_id);

        node->backward = [node, A, B, stream_id]() 
        {   
        A->grad = linG.MatMul(node->grad, lin.Transpose(B->output), stream_id);
        B->grad = linG.MatMul(lin.Transpose(A->output), node->grad, stream_id);

        };
        return node;
    }
    
graph GraphOperations::RelU(const graph& input)
    {   
        auto node = std::make_shared<Node>("intermediate");
        node->inputs = {input};
        node->output = ml.RelU(input->output);

        node->backward = [node, input]() 
        {   
            input->grad = lin.force_multiply(ml.deriv_ReLU(input->output), node->grad);
        };
        return node;
    }
    
graph GraphOperations::Softmax(const graph& input)
    {
        auto node = std::make_shared<Node>("intermediate");
        node->inputs = {input};
        node->output = ml.SoftMax(input->output);

        node->backward = [node, input]() 
        {
            input->grad = ml.deriv_SoftMax(input->output);
        };
        return node;
    }

graph GraphOperations::Softmask(const graph& input)
{
        auto node = std::make_shared<Node>("intermediate");
        node->inputs = {input};
        node->output = ml.SoftMask(input->output);

        node->backward = [node, input]() 
        {   
            input->grad = ml.deriv_SoftMask(input->output);
        };
        return node;
}

void GraphOperations::backprop(graph loss_node) 
{
    // Synchronize all GPU streams before starting backprop
    for (int i = 0; i < linG.stream_count(); ++i) {
        cudaStreamSynchronize(linG.get_stream(i));
    }
    
    // Rest remains the same as original CPU version
    // Properly initialize grad to match output shape
    loss_node->grad = {{1.0f}};
    graph_tree topo_order = topological_sort(loss_node);
    
    std::reverse(topo_order.begin(), topo_order.end());
    for (auto& node : topo_order) {
        if (node->backward) node->backward();
    }
}

void GraphOperations::clear_graph(graph loss_node) 
{
    // 1. Synchronize CUDA
    for (int i = 0; i < linG.stream_count(); ++i) 
        cudaStreamSynchronize(linG.get_stream(i));

    // 2. Get topo order (validate first!)
    auto topo_order = topological_sort(loss_node);
    if (topo_order.empty()) 
    {
        std::cerr << "Error: Empty graph!\n";
        return;
    }

    // 3. Clear in reverse order
    std::reverse(topo_order.begin(), topo_order.end());
    for (auto& node : topo_order) 
    {   
        if (!node) continue;  // Skip null nodes
        if(node->op_name == "intermediate")
        {
            node->backward = nullptr;  // Break cycles first
            node->output.clear();
            node->grad.clear();
            node->inputs.clear();
            node->op_name.clear();
            node.reset();
        }
    }
}

graph GraphOperations::MSE(const graph& pred, const Matrix_d& target) 
{
    auto node = std::make_shared<Node>("intermediate");
    node->inputs = { pred };
    node->output = Matrix_d(1, vector_d(1));
    node->grad = {{1.0f}};

    float sum = 0.0f;
    auto& y_pred = pred->output;
    for (size_t i = 0; i < y_pred.size(); ++i)
        for (size_t j = 0; j < y_pred[0].size(); ++j)
            sum += (y_pred[i][j] - target[i][j]) * (y_pred[i][j] - target[i][j]);

    node->output[0][0] = sum / (y_pred.size() * y_pred[0].size());

    node->backward = [node, pred, target]() 
    {   
        pred->grad.resize(pred->output.size(), vector_d(pred->output[0].size(), 0.0f));
        for (size_t i = 0; i < pred->output.size(); ++i)
            for (size_t j = 0; j < pred->output[0].size(); ++j)
                pred->grad[i][j] = 2.0f * (pred->output[i][j] - target[i][j]) / (pred->output.size() * pred->output[0].size());
    };

    return node;
}

graph GraphOperations::SoftmaxCrossEntropy(const graph& pred, const Matrix_d& target)
{
    auto node = std::make_shared<Node>("intermediate");
    node->inputs = {pred};

    // Forward: compute softmax first
    Matrix_d y_pred = ml.SoftMax(pred->output);
    node->output = {{1.0f}};
    float loss = 0.0f;
    for (size_t i = 0; i < y_pred.size(); i++)
    {
        for(int j = 0; j < y_pred[0].size(); j++)
        {   
            if( target[i][j] != 0)
            {
                loss -= target[i][j] * math.ln(y_pred[i][j]);
            } 
        }
    } 

    node->output[0][0] = loss;


    node->backward = [node, y_pred, pred, target]()
    {   
        pred->grad = Matrix_d(y_pred.size(), vector_d(y_pred[0].size(), 0.0f));
        for (int i = 0; i < y_pred.size(); i++)
        {
        for (int j = 0; j < y_pred[0].size(); j++)
        {
            pred->grad[i][j] = y_pred[i][j] - target[i][j];
        }

        }

    };

    return node;
}

graph GraphOperations::scale(graph input, float factor) 
    {
        auto node = std::make_shared<Node>("intermediate");
        node->inputs = {input};
        node->output = input->output;
        for (auto& row : node->output)
            for (auto& val : row)
                val *= factor;
                
        node->backward = [node, input, factor]() 
        {   
            input->grad = node->grad;
            for (auto& row : input->grad)
                for (auto& val : row)
                    val *= factor;
        };
        return node;
    }
    
graph GraphOperations::concatenate(const std::vector<graph>& inputs)  
{         
    auto node = std::make_shared<Node>("intermediate");         
    node->inputs = inputs;                
    
    // Store dimensions for backward pass
    std::vector<size_t> input_cols;
    size_t rows = inputs[0]->output.size();         
    size_t total_cols = 0;         
    
    for (auto& input : inputs) {
        size_t cols = input->output[0].size();
        input_cols.push_back(cols);
        total_cols += cols;
    }
    
    node->output = Matrix_d(rows, vector_d(total_cols, 0.0f));                  
    
    // Forward pass
    size_t current_col = 0;         
    for (size_t inp = 0; inp < inputs.size(); inp++) {             
        for (size_t i = 0; i < rows; i++) {                 
            for (size_t j = 0; j < input_cols[inp]; j++) {                     
                node->output[i][current_col + j] = inputs[inp]->output[i][j];                 
            }             
        }             
        current_col += input_cols[inp];         
    }         
    
    // Capture input_cols by value for the lambda
    node->backward = [node, inputs, input_cols, rows]() {           
        size_t current_col = 0;             
        
        for (size_t inp = 0; inp < inputs.size(); inp++) {
            auto& input = inputs[inp];
            
            // Ensure gradient matrix exists with correct size
            if (input->grad.empty()) {
                input->grad = Matrix_d(rows, vector_d(input_cols[inp], 0.0f));
            }
            
            // Accumulate gradients
            for (size_t i = 0; i < rows; i++) {                     
                for (size_t j = 0; j < input_cols[inp]; j++) {                         
                    input->grad[i][j] += node->grad[i][current_col + j];                    
                }                 
            }                 
            current_col += input_cols[inp];             
        }         
    };         
    return node; 
}

AttentionHead::AttentionHead(int id, int d_model, int d_k) : head_id(id) 
{   
    W_q = std::make_shared<AdamParameter>("head_" + std::to_string(id) + "_W_q", lin.RandMatrix(d_model, d_k, 2), 1e-3f);
    W_k = std::make_shared<AdamParameter>("head_" + std::to_string(id) + "_W_k", lin.RandMatrix(d_model, d_k, 2), 1e-3f);
    W_v = std::make_shared<AdamParameter>("head_" + std::to_string(id) + "_W_v", lin.RandMatrix(d_model, d_k, 2), 1e-3f);
}
    
graph AttentionHead::compute(graph input, GraphOperations& go, int head_id) 
{       
    const int num_streams = go.linG.stream_count() - 1;
    const int safe_stream_id = (head_id >= 0 && head_id < num_streams) ? head_id : 0;

    Math math;
    auto Q = go.matmul(input, W_q, safe_stream_id);
    auto K = go.matmul(input, W_k, safe_stream_id);
    auto V = go.matmul(input, W_v, safe_stream_id);
        
    auto K_transpose = go.transpose(K);
    auto scores = go.matmul(Q, K_transpose, safe_stream_id);
        
    // Scale scores by sqrt(d_k)
    auto scaled_scores = go.scale(scores, 1.0f / math.sqrt(W_k->output[0].size()));
    auto attention_weights = go.Softmax(scaled_scores);
    return go.matmul(attention_weights, V, safe_stream_id);
}

graph AttentionHead::compute_d(graph Decoder, graph encoder, GraphOperations& go, int head_id) 
{       
    const int num_streams = go.linG.stream_count() - 1;
    const int safe_stream_id = (head_id >= 0 && head_id < num_streams) ? head_id : 0;

    Math math;
    auto Q = go.matmul(Decoder, W_q, safe_stream_id);
    auto K = go.matmul(encoder, W_k, safe_stream_id);
    auto V = go.matmul(encoder, W_v, safe_stream_id);
        
    auto K_transpose = go.transpose(K);
    auto scores = go.matmul(Q, K_transpose, safe_stream_id);
        
    // Scale scores by sqrt(d_k)
    auto scaled_scores = go.scale(scores, 1.0f / math.sqrt(W_k->output[0].size()));
    auto attention_weights = go.Softmax(scaled_scores);
    return go.matmul(attention_weights, V, safe_stream_id);
}

MultiHeadAttention::MultiHeadAttention(int num_heads, int d_model) : d_model(d_model)
 {
        int d_k = d_model / num_heads;
        for (int i = 0; i < num_heads; i++) 
        {
            heads.emplace_back(i, d_model, d_k);
        }
        W_o = std::make_shared<AdamParameter>("W_o", lin.RandMatrix(d_model, d_model, 2), 1e-3f);

}
    
graph MultiHeadAttention::forward(graph input, GraphOperations& go) 
{   
    // std::cout << "Initializing forward vectors" <<std::endl;
    auto node = std::make_shared<Node>("forward");         
    node->inputs = {input};  
    int num_heads = heads.size();
    std::vector<std::future<graph>> futures;
    std::vector<graph> head_outputs(num_heads);
    
    // std::cout << "MHA sequential async execution" <<std::endl;
    // Launch each head in parallel with its own stream
    for (int i = 0; i < num_heads; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() 
        {
            return heads[i].compute(input, go, i);  // Stream ID = head index
        }));
    }
    
    // std::cout << "Getting sequential head of each asynchronous execution. Head: " << head_outputs.size() <<std::endl;
    // Collect results
    for (int i = 0; i < num_heads; ++i) 
    {
        head_outputs[i] = futures[i].get();
    }
    
    // std::cout << "Heads retrieved, synchronizing cuda streams \n";
    // Synchronize all streams before concatenation
    for (int i = 0; i < num_heads; ++i) {
        cudaStreamSynchronize(go.linG.get_stream(i));
    }
    // std::cout << "Synchronizing streams \n";
    
    // Final operations (CPU)
    // std::cout << "Concatenating heads" << std::endl;
    auto adjust = go.concatenate(head_outputs);
    // std::cout<< "Returning Matmul" <<std::endl;
    return go.matmul(adjust, W_o);
}

graph MultiHeadAttention::forward_d(graph Q_decoder, graph KV_encoder, GraphOperations& go) 
{   
    // std::cout << "Initializing forward decoder vectors" <<std::endl;
    auto node = std::make_shared<Node>("forward_d");         
    node->inputs = {Q_decoder, KV_encoder};  
    int num_heads = heads.size();
    std::vector<std::future<graph>> futures;
    std::vector<graph> head_outputs(num_heads);
    
    // std::cout << "MHA Decoder sequential async execution" <<std::endl;
    // Launch each head in parallel with its own stream
    for (int i = 0; i < num_heads; ++i) {
        futures.push_back(std::async(std::launch::async, [&, i]() 
        {
            return heads[i].compute_d(Q_decoder, KV_encoder, go, i);  // Stream ID = head index
        }));
    }
    
    // std::cout << "Getting sequential head of each asynchronous execution. Head: " << head_outputs.size() <<std::endl;
    // Collect results
    for (int i = 0; i < num_heads; ++i) 
    {
        head_outputs[i] = futures[i].get();
    }
    
    // Synchronize all streams before concatenation
    for (int i = 0; i < num_heads; ++i) {
        cudaStreamSynchronize(go.linG.get_stream(i));
    }
    
    // Final operations (CPU)
    // std::cout << "Concatenating heads" << std::endl;
    auto adjust = go.concatenate(head_outputs);
    // std::cout<< "Returning Matmul" <<std::endl;
    return go.matmul(adjust, W_o);
}

std::vector<std::shared_ptr<AdamParameter>> MultiHeadAttention::parameters() 
{
    std::vector<std::shared_ptr<AdamParameter>> params;
    for (auto& head : heads) 
    {
        params.push_back(head.W_q);
        params.push_back(head.W_k);
        params.push_back(head.W_v);
    }
    params.push_back(W_o);
    return params;
}
