#include <chrono>
#include "includes/DataLoader.h"
#include "includes/MathGPU.h"
#include "includes/backprop.h"
#include "includes/word2vec.h"

void Reading(text string)
{
    for (const auto & word: string)
        {
            std::cout << word << "\t";
        }
        std::cout << "\n";
        std::cout << "_____________________ \n";
}

int main() 
{
    srand(static_cast<unsigned>(time(nullptr)));
    str folderpath = "Enter your folder path";
    GraphOperations go;
    Linear_Algebra lin;
    Math math;
    Machine_Learning ml;
    text words = Load(folderpath);
    text cleanedWords = read_words(words, 0, 50000);
    // Reading(cleanedWords);

    int length = 8; // Length of the text generated during inference
    int context_size = 32; // Size of the context window of inference
    int n = 128;  // Size of context window of training loop
    int embed = 128; // Size of vector embedding or d_model
    int uniform = 1; // Size of uniform distribution [-uniform, uniform]
    int topk = 5; // Size of topK inference probability, 1 is argmax
    int p_val = 256; 
    int t_val = 256;
    Tokenization T = tokenizer(cleanedWords, embed);
    MultiHeadAttention mha_enc(8, embed);
    MultiHeadAttention mha_dec_self(8, embed);
    MultiHeadAttention mha_dec_cross(8, embed);
    int layer_size = T.word_map.size();

    // Define parameters with correct dimensions   
    auto W1 = std::make_shared<AdamParameter>("W1", lin.RandMatrix(embed, p_val, uniform), 1e-1f);  // d_model x p
    auto b1 = std::make_shared<AdamParameter>("b1", lin.RandMatrix(1, p_val, uniform), 1e-1f);   // 1 x p
    auto W2 = std::make_shared<AdamParameter>("W2", lin.RandMatrix(p_val, embed, uniform), 1e-1f);  // p x d_model
    auto b2 = std::make_shared<AdamParameter>("b2", lin.RandMatrix(1, embed, uniform), 1e-1f);   // 1 x d_model
    auto W1_D = std::make_shared<AdamParameter>("W1_D", lin.RandMatrix(embed, t_val, uniform), 1e-1f);  // d_model x t
    auto b1_D = std::make_shared<AdamParameter>("b1_D", lin.RandMatrix(1, t_val, uniform), 1e-1f);   // 1 x t
    auto W2_D = std::make_shared<AdamParameter>("W2_D", lin.RandMatrix(t_val, embed, uniform), 1e-1f);  // t x d_model
    auto b2_D = std::make_shared<AdamParameter>("b2_D", lin.RandMatrix(1, embed, uniform), 1e-1f);   // 1 x d_model
    auto W3 = std::make_shared<AdamParameter>("W3", lin.RandMatrix(embed, layer_size, uniform), 1e-1f);  //d_model x Layer_size
    auto b3 = std::make_shared<AdamParameter>("b3", lin.RandMatrix(1, layer_size, uniform), 1e-1f);   // 1 x layer_size

    for(int z = 0; z < 10000; z++)
    {  
        std::cout << "Epoch: " << z * 1 << "\n";
        auto start = std::chrono::high_resolution_clock::now();
        int random_elem = math.random_int(0, layer_size - n - 2);
        text stories = read_words(cleanedWords, random_elem, random_elem + n);
        text initial_stories = stories;
        initial_stories.insert(initial_stories.begin(), "<start>");
        Matrix_d input_encoding = encoder_input(T.word_map, stories);
        Matrix_d input_decoding = decoder_input(T.word_map, stories);
        stories.push_back("<end>");
        Matrix_d target_data = one_hot_encoding(T.index, stories);
        auto input_enc = std::make_shared<Node>("input_encoding");
        input_enc->output = PEncoding(input_encoding);
        auto input_dec = std::make_shared<AdamParameter>("Input decoder", PEncoding(input_decoding), 1e-1f);
        std::vector<std::shared_ptr<AdamParameter>> parameters = {input_dec, W1, W2, W3, W1_D, W2_D, b1, b2, b3, b1_D, b2_D};
        input_encoding.clear();
        input_decoding.clear();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        if (z % 250  == 0)
        {   
            std::cout << "Time taking for all variable initialization: " << elapsed.count() << "s \n";
        }

        for(int epoch = 0; epoch < 1; epoch++)
        {   
            auto start2 = std::chrono::high_resolution_clock::now();
            auto QKV_enc = mha_enc.forward(input_enc, go); // Forward(n x d_model) = n x d_model
            auto enc_add1 = go.add(input_enc, QKV_enc); // n x d_model + n x d_model = n x d_model
            auto enc_norm1 = go.LayerNorm(enc_add1);  // LN(n x d_model) = n x d_model
            auto enc_fc1 = go.matmul(enc_norm1, W1); // n x d_model x d_model  x p_val = n x p_val
            auto enc_fc1_b = go.add(enc_fc1, b1); // n x p_val + n x p_val = n x p_val
            auto enc_relu = go.RelU(enc_fc1_b);  // Rel(n x p_val) = n x p_val
            auto enc_fc2 = go.matmul(enc_relu, W2); // n x p_val x p_val x d_model = n x d_model
            auto enc_fc2_b = go.add(enc_fc2, b2); // n x d_model + n x d_model = n x d_model
            auto enc_add2 = go.add(enc_fc2_b, enc_norm1); // n x d_model + n x d_model  = n x d_model
            auto enc_out = go.LayerNorm(enc_add2); // LN(n x d_model) =  n x d_mode
            auto QKV_dec_self = mha_dec_self.forward(input_dec, go); // forward(n+1 x d_model) =  n+1 x d_model
            auto dec_add1 = go.add(input_dec, QKV_dec_self); // n+1 x d_model + n+1 x d_model  =  n+1 x d_model
            auto dec_norm1 = go.LayerNorm(dec_add1); // LN(n+1 x d_model) =  n+1 x d_model
            auto QKV_dec_cross = mha_dec_cross.forward_d(dec_norm1, enc_out, go); // forward(n+1 x d_model, n x d_model) =  n+1 x d_model
            auto dec_add2 = go.add(dec_norm1, QKV_dec_cross); // n+1 x d_model + n+1 x d_model =  n+1 x d_model
            auto dec_norm2 = go.LayerNorm(dec_add2); // LN(n+1 x d_model) = n+1 x d_model
            auto dec_fc1 = go.matmul(dec_norm2, W1_D); // n+1 x d_model x d_model x t = n+1 x t
            auto dec_fc1_b = go.add(dec_fc1, b1_D); // n+1 x t + n+1 x t = n+1 x t
            auto dec_relu = go.RelU(dec_fc1_b);  // Rel(n+1 x t) = n+1 x t
            auto dec_fc2 = go.matmul(dec_relu, W2_D); // n+1 x t x t x d_model = n+1 x d_model
            auto dec_fc2_b = go.add(dec_fc2, b2_D);  // n+1 x d_model + n+1 x d_model = n+1 x d_model
            auto dec_relu2 = go.RelU(dec_fc2_b);  // Rel(n+1 x t) = n+1 x d_model
            auto dec_add3 = go.add(dec_relu2, dec_norm2);  // n+1 x d_model + n+1 x d_model = n+1 x d_model
            auto dec_out_norm = go.LayerNorm(dec_add3);  // LN(n+1 x k) = n+1 x d_model
            auto final_logits = go.matmul(dec_out_norm, W3);  // n+1 x d_model x d_model x dictionary = n+1 x dictionary
            auto final_out= go.add(final_logits, b3); // (n+1) x dictionary + (n+1) x dictionary = n+1 x dictionary
            auto loss = go.SoftmaxCrossEntropy(final_out, target_data); // Sft(n+1 x dictionary, n+1 x dictionary) = scalar
            go.backprop(loss);
            for (auto& p : parameters)
            {   
                p->update();
            }

            auto end2 = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed2 = end2 - start2;
            if (epoch % 250 == 0 && z % 250 == 0)
            {   
                lin.printMatrixHead(target_data);
                lin.printMatrixHead(ml.SoftMax(final_out->output));
                std::cout << "Time taking for forward and backward pass: " << elapsed2.count() << "s \n";
                std::cout << "Epoch " << epoch << " Loss: " << loss->output[0][0] << "\n";
                std::cout << "Number of Z steps: " << z << "\n";
                vec_replace(T.word_map, initial_stories, input_dec->output);
            }

            go.clear_graph(loss);

            if (epoch % 10 == 0 && z % 250 == 0) // inference
            {
                std::cout << "Inference at Z loop : " <<z << "\n";
                auto inf_start = std::chrono::high_resolution_clock::now();
                text textinference = read_words(cleanedWords, random_elem, random_elem + context_size);
                text textinference_d  =  textinference;
                text output_print = textinference;
                std::cout << "Beginning the inference \n";
                for(int i = 0; i < length; i++)
                {
                    Matrix_d input_encoding = encoder_input(T.word_map, textinference);
                    Matrix_d input_decoding = decoder_input(T.word_map, textinference_d);
                    Matrix_d inf_enc = PEncoding(input_encoding);
                    Matrix_d inf_dec = PEncoding(input_decoding);
                    input_encoding.clear();
                    input_decoding.clear();
                    auto inference_enc = std::make_shared<Node>("inferential encoder");
                    auto inference_dec = std::make_shared<Node>("differential encoder");
                    inference_enc->output = inf_enc;
                    inference_dec->output = inf_dec;
                    auto inf_QKV_enc = mha_enc.forward(inference_enc, go); 
                    auto inf_enc_add1 = go.add(inference_enc, inf_QKV_enc); 
                    auto inf_enc_norm1 = go.LayerNorm(inf_enc_add1);  
                    auto inf_enc_fc1 = go.matmul(inf_enc_norm1, W1);
                    auto inf_enc_fc1_b = go.add(inf_enc_fc1, b1);
                    auto inf_enc_relu = go.RelU(inf_enc_fc1_b);  
                    auto inf_enc_fc2 = go.matmul(inf_enc_relu, W2);
                    auto inf_enc_fc2_b = go.add(inf_enc_fc2, b2); 
                    auto inf_enc_add2 = go.add(inf_enc_fc2_b, inf_enc_norm1); 
                    auto inf_enc_out = go.LayerNorm(inf_enc_add2);
                    auto inf_QKV_dec_self = mha_dec_self.forward(inference_dec, go); 
                    auto inf_dec_add1 = go.add(inference_dec, inf_QKV_dec_self);
                    auto inf_dec_norm1 = go.LayerNorm(inf_dec_add1);
                    auto inf_QKV_dec_cross = mha_dec_cross.forward_d(inf_dec_norm1, inf_enc_out, go);
                    auto inf_dec_add2 = go.add(inf_dec_norm1, inf_QKV_dec_cross); 
                    auto inf_dec_norm2 = go.LayerNorm(inf_dec_add2);

                    // Decoder Feed Forward:
                    auto inf_dec_fc1 = go.matmul(inf_dec_norm2, W1_D); 
                    auto inf_dec_fc1_b = go.add(inf_dec_fc1, b1_D); 
                    auto inf_dec_relu = go.RelU(inf_dec_fc1_b);
                    auto inf_dec_fc2 = go.matmul(inf_dec_relu, W2_D);
                    auto inf_dec_fc2_b = go.add(inf_dec_fc2, b2_D); 
                    auto inf_dec_relu2 = go.RelU(inf_dec_fc2_b);

                // Output Projection:
                    auto inf_dec_add3 = go.add(inf_dec_relu2, inf_dec_norm2); 
                    auto inf_dec_out_norm = go.LayerNorm(inf_dec_add3);
                    auto inf_final_logits = go.matmul(inf_dec_out_norm, W3);
                    auto inf_final_out= go.add(inf_final_logits, b3); // (n+1) x dictionary + (n+1) x dictionary = n+1 x dictionary
                    auto probs = go.Softmax(inf_final_out);
                    vector_d infprobs = probs->output[probs->output.size() - 1];
                    int idx = math.TopKSampler(infprobs, topk);
                    if( i % 10 == 0)
                    {
                        std::cout << "The inference probability is: " << infprobs[idx] << "\n";
                    }
                    infprobs.clear();
                    go.clear_graph(probs);
                    if (i == length - 1)
                    {
                        std::cout << " The index of last predicted word: " << idx << " \n";
                    }
                    
                    for (const auto& pair : T.index)
                    {   
                        if(pair.second == idx)
                        {   
                            output_print.push_back(pair.first);
                            textinference.push_back(pair.first);
                            textinference_d.push_back(pair.first);
                            break;
                        }
                    }
                }
                
                for (const auto & word: output_print)
                {
                    std::cout << word << "\t";
                }
                std::cout << "\n";
                std::cout << "_____________________ \n";
        
                auto inf_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<float> inf_elapsed = inf_end - inf_start;
                
                std::cout << "Time elapsed for inference " << length << " times: " << inf_elapsed.count() << "s" << "\n";
                textinference.clear();
                textinference_d.clear();
                output_print.clear();
                }

        }
    }
    return 0;
}




