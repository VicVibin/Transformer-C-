#pragma once
#include "includes/engine.h"
#include "includes/dataloader.h"

#include <cerrno>
#include <cstring>  
#include <sys/stat.h>
#include <filesystem>
#if defined(_WIN32)
  #include <direct.h> 
#endif


class LLM
{
private:
    const int embed_dim;
    TextualEmbedding& embedder;
    DataLoading Dataload;
    SingleHeadAttention T1, T2, T3;
    TimeMLPBlock fc1, fc2;
    Linear proj;
    
 
public:
    GraphOperations go;
    LLM(GraphOperations& go_ref, TextualEmbedding& embed, const Text& Database,int batch, int clen, int hidden_dim=128) :
    go(go_ref), embedder(embed), embed_dim(embed.embed_dim), Dataload(embed, Database, batch, clen),
    T1(go, embed_dim, hidden_dim),
    T2(go, embed_dim, hidden_dim),
    T3(go, embed_dim, hidden_dim),
    fc1(go,embed_dim, embed_dim), 
    fc2(go,embed_dim, embed_dim), proj(go, embed_dim, embedder.Vocabulary.size(), "Projection")
    {
        if(embedder.Vocabulary.size()  == 0)
        {
            std::cerr << "You may have forgotten to call embedder.updateVocabulary(data) outside the LLM Constructor. \n";
            std::cerr << "Please update the vocabulary with your dataset before initializing the LLM. \n";
            std::exit(1);
        }
        if (embedder.Vocabulary.size() <= clen)
        {
            std::cerr << "Warning: Vocabulary size is less than or equal to context length. This may lead to issues with training. \n";
        }

    }

    std::pair<graph,graph> build_train(const BatchTexts& data)
    {
        auto encoder_embeds = Dataload.forward(data, "E");
        auto decoder_embeds = Dataload.forward(data, "D");
        auto target = Dataload.forward(data, "T");

        auto pos_encodded = go.MatrixPositionalEncoding(encoder_embeds); 
        auto Att1 = T1.forward(pos_encodded); 
        auto A1 = fc1.forward(Att1); 

        auto AN1 = go.LayerNorm(go.Add(A1,Att1));
        auto pos_decoded = go.MatrixPositionalEncoding(decoder_embeds);
        auto Att2 = T2.forward(pos_decoded); 
        auto CrossAtt = T3.cross_forward(Att2, AN1);


        auto A2 = fc2.forward(CrossAtt);
        auto AN2 = go.LayerNorm(go.Add(A2,CrossAtt));

        // Projection
        auto logits = proj.forward(AN2);
        auto loss = go.SoftMaxCrossEntropy(logits, target);
        go.nodes = topological_sort(loss);
        return {encoder_embeds, decoder_embeds};
    }

    void generate(const Text& prompt, const int max_len = 0)
    {
        graph encoder_embeds = std::make_shared<NodeBackProp>("Encoder Embeddings", 1, 1, prompt.size(), embed_dim, 1);
        graph decoder_embeds = std::make_shared<NodeBackProp>("Decoder Embeddings", 1, 1, 1, embed_dim, 1);

        int* indexpointer; 
        SafeCudaMalloc("ArgMax Pointer", indexpointer, 1);
        embedder.encodeText(prompt, "E");
        embedder.forward(encoder_embeds, "E");

        embedder.encodeText({"<START>"}, "D");
        decoder_embeds->dim[2] = 1; 
        decoder_embeds->total = embed_dim;
        embedder.forward(decoder_embeds, "D");

        // Encoder ================ //
        auto pos_encodded = go.MatrixPositionalEncoding(encoder_embeds);
        auto Att1 = T1.forward(pos_encodded);
        auto A1 = fc1.forward(Att1);
        auto AN1 = go.LayerNorm(go.Add(A1,Att1));
        auto K_enc = T3.k->forward(AN1);
        auto V_enc = T3.v->forward(AN1);
        // ======================= //

        go.nodes = topological_sort(AN1);
        go.forward();

        // Initializing KV_Cache ====//
        K_enc->forward();
        V_enc->forward();
        KVCache kv_T2(embedder.MAX_CONTEXT_LEN, T2.hidden);
        KVCache kv_T3(embedder.MAX_CONTEXT_LEN, T3.hidden);
        kv_T3.current_len = V_enc->dim[2];
        cudaMemcpy(kv_T3.K,K_enc->output,K_enc->total*sizeof(float),cudaMemcpyDeviceToDevice);
        cudaMemcpy(kv_T3.V,V_enc->output,V_enc->total*sizeof(float),cudaMemcpyDeviceToDevice);
        // ==========================//

        go.clear_graph();
        K_enc->free();
        V_enc->free();
        int start_idx = 0;
        
        std::cout << "Generating text: \n" <<  "\n";
        for (const auto& word : prompt) std::cout << word << "\t";
        while (true)
        {
            auto Att2 = T2.cached_forward(decoder_embeds, kv_T2, start_idx);
            auto CrossAtt = T3.cached_cross_forward(Att2, kv_T3); 
            auto A2 = fc2.forward(CrossAtt);
            auto AN2 = go.LayerNorm(go.Add(A2, CrossAtt));
            auto logits = proj.forward(AN2);

            go.nodes = topological_sort(logits);
            go.forward();

            int next_token = TopKSampleToCPU(logits, indexpointer, 10);
            const str next_word = embedder.KeySpace[next_token];
            if (next_word == "<end>" || next_word == "<pad>" || kv_T2.current_len >= embedder.MAX_CONTEXT_LEN) break;
            std::cout << next_word << "\t ";
            if (max_len != 0 && kv_T2.current_len >= max_len) break;

            embedder.rencodeText({next_word}, "D");
            embedder.rforward(decoder_embeds, "D");
            go.clear_graph();
            start_idx++;
        }
        std::cout << "\n \n";
        kv_T2.free();
        kv_T3.free();
        cudaFree(indexpointer);
        decoder_embeds->clear();
    }
    
    void train(const int num_batches, const int percent = 1, const float min_loss = 1e-2f)
    {
        
        BatchTexts batch_data;
        

        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
        {
            
            Timing forw("Forward"), back("Backward");
            batch_data = Dataload.load_data(); 
            auto embs = build_train(batch_data); 
            if (batch_idx == 0) {printf("Memory Requirements: %.3f GB \n", go.GB); go.GB = 0.0f;}  
            go.zero_grad();

            if (batch_idx % percent == 0)
            {   
                forw.start(); go.forward(); forw.end(); 
                std::cout << "Batch " << batch_idx+1 << "/" << num_batches << ", Loss: " << go.loss << "\n";
                back.start(); go.backward(); back.end();
                go.ParameterUpdate(); 
                go.clear_graph();
                generate({"Jesus"}, 20); 
                continue;
            }
            if (go.loss < min_loss) break;

            else
            {
                go.forward();
                go.backward();
                //embedder.EmbeddingUpdate(embs.first, "E");
                //embedder.EmbeddingUpdate(embs.second,"D");
                go.ParameterUpdate(); 
            }
            go.clear_graph();
        }   
        
        
    }

    
    void save(const str& filename) const
    {
        std::ofstream f(filename, std::ios::binary);
        if (!f) {
            std::cerr << "Error opening file for saving: " << filename << std::endl;
            std::exit(1);
        }
        T1.save(f); T2.save(f); T3.save(f);
        fc1.save(f); fc2.save(f);
        proj.save(f);
        f.close();
        std::cout << "Model saved successfully to " << filename << "\n";
    
    }

    void load(const str& filename)
    {
        std::ifstream f(filename, std::ios::binary);
        if (!f) {
            std::cerr << "Error opening file for loading: " << filename << std::endl;
            std::exit(1);
        }
        T1.load(f); T2.load(f); T3.load(f); fc1.load(f); fc2.load(f); proj.load(f);
        f.close();
        std::cout << "Model loaded successfully from " << filename << "\n";
    }
        
};


class ResidualBlock
{
public:
    GraphOperations &go;
    Convolute2D *conv1, *conv2, *skipconv;
    Identity *skip;
    Linear *time_mlp;
    int in, out, hidden, stride;

    ResidualBlock(GraphOperations &go_ref, const int in_channels, const int out_channels, const int t_hidden, const int stride=1): 
    go(go_ref), in(in_channels), out(out_channels), hidden(t_hidden), stride(stride)
    {
        conv1 = new Convolute2D(go, in_channels,  out_channels,3,3,stride,1, "ResBlock Conv1");
        conv2 = new Convolute2D(go, out_channels, out_channels,3,3,1,1, "ResBlock Conv2");
        time_mlp = new Linear(go, t_hidden, out_channels, "ResBlock Time MLP");
        skipconv = new Convolute2D(go,in_channels, out_channels,1,1,stride,0, "ResBlock SkipConv");
        skip = new Identity(go, "Resblock Identity SkipConv");
        
    }

    void save(std::ofstream& f) const
    {
        conv1->save(f);
        conv2->save(f);
        skipconv->save(f);
        time_mlp->save(f);
    }

    void load(std::ifstream& f)
    {
        conv1->load(f);
        conv2->load(f);
        skipconv->load(f);
        time_mlp->load(f);
    }

    graph forward(const graph& x, const graph & t_emb)
    {
        auto h = go.SILU(go.GroupNorm(conv1->forward(x)));
        auto time = time_mlp->forward(t_emb);
        h = go.Broadcast_Channel(h, time);
        h = go.GroupNorm(conv2->forward(h));
        auto skipnet = (in != out || stride != 1) ? skipconv->forward(x) : skip->forward(x);
        return go.SILU(go.Add(h, skipnet));
    }
};

class U_NET
{
public:
    GraphOperations &go;
    const graph &input, &target, &text_embed;
    graph prediction;
    double* global_scale;
    int t_embed_dim, t_hidden, t, batch;

    ResidualBlock *enc1, *enc2, *enc3, *enc4, *b0, *b1, *dec1, *dec2, *dec3;
    Convolute2D *out;
    Convolute2DT *up1, *up2, *up3;
    TimeMLPBlock *time_mlp;
    float loss;
    
    U_NET(GraphOperations& goref,const graph& input, const graph& target, const graph& text_embed, 
    const int in_channels=3,const int out_channels=3, const int init_depth=64,const int t_node=32): 
    batch(input->dim[0]), go(goref), t_embed_dim(t_node), t_hidden(t_node/2), 
    input(input), target(target), text_embed(text_embed)
    
    {
    if(SHOULDNORM) 
    {
        SafeCudaMalloc("Global Scale", global_scale, 1); 
        fillKernel<<<1,1>>>(global_scale, 0.0, 1);
        cudaDeviceSynchronize(); 
        //CheckError("Fill Kernel for Global Scale");
    }
        
    time_mlp = new TimeMLPBlock(go, t_embed_dim, t_hidden);
    enc1 = new ResidualBlock(go, in_channels, init_depth, t_hidden);
    enc2 = new ResidualBlock(go, init_depth, init_depth * 2, t_hidden, 2);
    enc3 = new ResidualBlock(go, init_depth * 2, init_depth * 4, t_hidden, 2);
    enc4 = new ResidualBlock(go, init_depth * 4, init_depth * 8, t_hidden, 2);

    b0 = new ResidualBlock(go, init_depth * 8, init_depth * 8, t_hidden);
    b1 = new ResidualBlock(go, init_depth * 8, init_depth * 8, t_hidden);

    up1 = new Convolute2DT(go, init_depth * 8, init_depth * 8, 2, 2, 2, 0, "UpConv 1");
    dec1 = new ResidualBlock(go, init_depth * 8 + init_depth * 4, init_depth * 4, t_hidden);

    up2 = new Convolute2DT(go, init_depth * 4, init_depth * 4, 2, 2, 2, 0, "UpConv 2");
    dec2 = new ResidualBlock(go, init_depth * 4 + init_depth * 2, init_depth * 2, t_hidden);

    up3 = new Convolute2DT(go, init_depth * 2, init_depth * 2, 2, 2, 2, 0, "UpConv 3");
    dec3 = new ResidualBlock(go, init_depth * 2 + init_depth, init_depth, t_hidden);

    out = new Convolute2D(go, init_depth, out_channels, 1,1,1,0, "Output Convolution");
    }

    void save(const str& filename) const
    {

        std::filesystem::path path(filename);
        std::filesystem::path dir = path.parent_path();

        if (!dir.empty())
        {
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            if (ec)
            {
                std::cerr << "Failed to create directory: " << dir << "\n";
                std::exit(1);
            }
        }

        str tmp = filename + ".tmp";

        std::ofstream f(tmp, std::ios::binary);
        if (!f)
        {
            std::cerr << "Error opening temp file: " << tmp << "\n";
            std::exit(1);
        }

        time_mlp->save(f);
        enc1->save(f); enc2->save(f); enc3->save(f); enc4->save(f);
        b0->save(f); b1->save(f);
        dec1->save(f); dec2->save(f); dec3->save(f);
        up1->save(f); up2->save(f); up3->save(f);
        out->save(f);
        

        if (!f.good())
        {
            std::cerr << "Error during write to temp file\n";
            f.close();
            std::filesystem::remove(tmp);
            std::exit(1);
        }

   
        f.flush();
        f.close();

        std::error_code ec;

    std::filesystem::rename(tmp, filename, ec);

    if (ec)
    {
        std::error_code remove_ec;
        std::filesystem::remove(filename, remove_ec);

        std::filesystem::rename(tmp, filename, ec);

        if (ec)
        {
            std::cerr << "Failed to rename temp file to final: " << filename << "\n";
            std::cerr << "Temp file left at: " << tmp << "\n";
            std::exit(1);
        }
    }

    std::cout << "Model saved successfully to " << filename << "\n";

    }

    void load(const str& filename)
    {
        std::ifstream f(filename, std::ios::binary);
    if (!f){
        std::cerr << "Error opening file for loading: " << filename << "\n";
        std::exit(1);
        }

    time_mlp->load(f);
    enc1->load(f); enc2->load(f); enc3->load(f); enc4->load(f);
    b0->load(f); b1->load(f);
    dec1->load(f); dec2->load(f); dec3->load(f);
    up1->load(f); up2->load(f); up3->load(f);
    out->load(f);

    f.peek();
    if (!f.eof())
        std::cerr << "Warning: file '" << filename << "' has leftover bytes after loading — "
                  << "save/load structure may be mismatched\n";

    f.close();
    std::cout << "Model loaded successfully from " << filename << "\n";
    }

    void build_train()
    {
        auto t_emb = go.PositionalEncoding(t,t_embed_dim);   
        auto t_mlp = time_mlp->forward(t_emb); 

        auto e1 = enc1->forward(input, t_mlp);  
        auto e2 = enc2->forward(e1, t_mlp); 
        auto e3 = enc3->forward(e2, t_mlp);     
        auto e4 = enc4->forward(e3, t_mlp);

        // Attention
        auto b_0  = b0->forward(e4, t_mlp);
        auto b_1  = b1->forward(b_0, t_mlp);
        
        // FIX: Upsample THEN concatenate
        auto up_1 = up1->forward(b_1);   auto d1_in = go.CopyCrop(up_1, e3);  auto d1 = dec1->forward(d1_in, t_mlp); 
        auto up_2 = up2->forward(d1);    auto d2_in = go.CopyCrop(up_2, e2);  auto d2 = dec2->forward(d2_in, t_mlp);
        auto up_3 = up3->forward(d2);    auto d3_in = go.CopyCrop(up_3, e1);  auto d3 = dec3->forward(d3_in, t_mlp);
        auto logits = out->forward(d3);

        auto loss = go.MeanSquaredError(logits, target);
        go.nodes = topological_sort(loss);
        prediction = logits;

        

    }

    void build_inference(const graph& test_input)
    {
        auto t_emb = go.PositionalEncoding(t,t_embed_dim);   
        auto t_mlp = time_mlp->forward(t_emb); 

        auto e1 = enc1->forward(test_input, t_mlp);  
        auto e2 = enc2->forward(e1, t_mlp); 
        auto e3 = enc3->forward(e2, t_mlp);     
        auto e4 = enc4->forward(e3, t_mlp);

        auto b_0  = b0->forward(e4, t_mlp);
        auto b_1  = b1->forward(b_0, t_mlp);
        
        auto up_1 = up1->forward(b_1);   auto d1_in = go.CopyCrop(up_1, e3);  auto d1 = dec1->forward(d1_in, t_mlp); 
        auto up_2 = up2->forward(d1);    auto d2_in = go.CopyCrop(up_2, e2);  auto d2 = dec2->forward(d2_in, t_mlp);
        auto up_3 = up3->forward(d2);    auto d3_in = go.CopyCrop(up_3, e1);  auto d3 = dec3->forward(d3_in, t_mlp);
        prediction = out->forward(d3);
        go.nodes = topological_sort(prediction);
    }

    void zero_grad(){go.zero_grad();}
    void forward(){go.forward(); loss = go.loss;}
    void backward(){ go.backward();if(SHOULDNORM) {Zerograd("Global Scale", global_scale, 1); go.accumulate(global_scale); go.clipNorm(global_scale);}}
    void parameterUpdate(){go.ParameterUpdate();}
    void printvals(){for (const auto&node: go.nodes) printHeadGPU(node);}   
    void printgrads(){for (const auto&node: go.nodes) printHeadGPU(node, 1);}
    void printparams(){for (const auto&node: go.nodes) if(node->printparams) printHeadGPU(node);}
    
    void train()
    {
        Timing forw("Forward"), back("Backward");
        zero_grad(); forward(); 
         backward(); parameterUpdate();
    }
};

template<typename NET>
class Sampling{
public:
    NET &model;
    int T, t;
    const graph &input;
    graph u_theta;
    std::random_device rd;

    Sampling(NET& trained_model, const graph &noisy_image, int T_in,  int Big_T_in): 
    model(trained_model), input(noisy_image), t(T_in) ,T(Big_T_in)
    {   
        std::cout << "Building Inference Model for Langevin Sampling \n";
        model.build_inference(input);
        std::cout << "Final loss before sampling: " << model.loss << "\n";
        model.prediction->op_name = "Model";
        const int a = input->dim[0], b = input->dim[1], c = input->dim[2], d = input->dim[3];
        u_theta = std::make_shared<NodeBackProp>("U0(X_t,t)",a,b,c,d,1);
        if(input->dim[0] != 1) std::cout << "Multi Image inference for Langevin.... \n"; 
    
    }
    
    void forward(double s = 0.008)
    {   
        model.t = t;
        model.zero_grad();
        model.forward();
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
        diffuse(input->output, model.prediction->output, u_theta->output, input->total, t, T, s, seed);--t;

    }

    void display(const int row=0, const int col=0)
    {   
        StandardDeNorm(input);
        cv::Mat img = n2i(input);
        if (row > 0 && col > 0) cv::resize(img, img, cv::Size(col, row), 0, 0, cv::INTER_AREA);
        cv::imwrite("final_sample.png", img);
        cv::imshow("Final Sample", img);
        cv::waitKey(0);
    }

    void loop(const int start, const int till = 0, const int par = 100)
    {
        for(int i = start; i > till; --i)
        {
            if (i % par == 0) printf("Step: %i/%i \n", i, T);
            forward();
        }}
};

/*

int main(){
    const int MAX_VOCAB_SIZE = 25000;
    const int MAX_BATCH_SIZE = 16;
    const int MAX_CONTEXT_LEN = 16384;
    const int EMBED_DIM = 128;
    const int context_len = 32;
    const int HIDDEN_DIM = 256;

    GraphOperations go;
    TextualEmbedding embedder(EMBED_DIM,MAX_BATCH_SIZE, MAX_CONTEXT_LEN, MAX_VOCAB_SIZE);
    Text Db = LoadStory("C:/Users/victo/Documents/Coding_Projects/text");
    Text Database = read_words(Db, 0, Db.size());
    embedder.updateVocabulary(Database);
    printf("Total Vocabulary Size: %i | Total Word Count: %i \n", embedder.Vocabulary.size(), Database.size());
    LLM model(go, embedder, Database, MAX_BATCH_SIZE, context_len, HIDDEN_DIM);
    model.train(20000, 100);
    model.generate({"so","sherlock", "holmes", "said"}, 30);
}


int main()
{
    GraphOperations go;
    const int T = 1000, init_depth = 64, t_hidden = 128, img_size = 64, epochs = 10000;
    int t;
    std::random_device rd;
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, T-1);

    auto base = Bi2n("KPOP", 1, img_size, img_size);
    StandardNorm(base);
    auto input  = go.like(base, "Input Image"), target = go.like(base, "Target Image");

    std::cout << "Building U-NET model \n";
    U_NET model(go, input, target, nullptr, 3, 3, init_depth, t_hidden);
    model.load("best_models/huntrix.bin");

    const bool train = false;
    if (train){
    model.build_train();
    std::cout << "Starting Training Loop for " << epochs << " epochs \n";
    for(int epoch = 0; epoch < epochs; ++epoch)
    {
        model.t = t = dist(gen); 
        prepare(base, input, target, t, T, ((uint64_t)rd() << 32) | rd());
        if (epoch % 5 == 0) go.calculate_loss = true;
        model.train();
        if(epoch % 5 == 0)
        {   
            printf("Epoch %i, Loss: %f at t = %i \n", epoch+1, model.loss, t); 
            go.calculate_loss = false;
        }    
        if (epoch % 100 == 0) model.save("gmodel"+std::to_string(epoch)+".bin");
    }
    go.clear_graph();
    }
    
    input->clear();
    target->clear();
    auto test = std::make_shared<NodeBackProp>("Test", 1, base->dim[1], base->dim[2], base->dim[3],1);
    Noise(test);
    Sampling<U_NET> sampler(model, test, T-1,T);
    sampler.loop(T-1, 0);
    //BPrintImage(base, 512, 512);
    sampler.display(512,512);
    return 0;

}