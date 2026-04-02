#include "includes/engine.h"

void isNan(const str name, const float* X, const long long total)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (total + tpb-1)/tpb;
    ISNAN<<<bpg,tpb>>>(X, total);
    CheckError(name + "   has nan value");
    return;
}

NodeBackProp::NodeBackProp(str name, int batch, int d1, int d2, int d3, int allocation) : op_name(name)
{
        const int tpb = THREADSPERBLOCK;
        dim[0] = batch;
        dim[1] = d1;
        dim[2] = d2;
        dim[3] = d3;
        total = batch * d1 * d2 * d3;
        
        if(allocation == 1)
        {

            const int bpg = (total+tpb-1)/tpb;
            SafeCudaMalloc(op_name, output, total);
            SafeCudaMalloc(op_name, grad, total);
            CheckError("Scale initialization");

            fillKernel<<<bpg,tpb>>>(grad,0.0f, total);
            CheckError("Fill Kernel for Gradient");
            
        }

        else
        {
            output = nullptr;
            grad = nullptr;
        }
}

void NodeBackProp::clear()
{
    cudaFree(output);
    cudaFree(grad);
    inputs.clear();
    op_name.clear();
}

void NodeBackProp::reshape(std::vector<int> new_dims)
{
    if(new_dims.size() !=4)
    {
        std::cerr << "Reshape only supports 4D tensors\n";
        std::exit(1);
    }
    
    int new_total = new_dims[0]*new_dims[1]*new_dims[2]*new_dims[3];
    
    if(new_total != total)
    {std::cerr << "Reshape cannot change total size of tensor\n";
    std::exit(1);}
    for(int i=0;i<4;++i){dim[i] = new_dims[i];}
}

Ipointer i2p(const std::string& filepath, int row_size, int col_size) 
{
    Ipointer result;
    
    // Read image in color
    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to load image!\n";
        std::exit(1);
    }

    // If resizing is requested
    if (row_size > 0 && col_size > 0) {
        cv::resize(img, img, cv::Size(col_size, row_size), 0, 0, cv::INTER_AREA);
    }

    // Allocate memory for channels × rows × cols
    int channels = img.channels();
    int rows = img.rows;
    int cols = img.cols;
    float* P = (float*)malloc(rows * cols * channels * sizeof(float));

    // Fill in channel-first format (C × H × W)
    for (int ch = 0; ch < channels; ch++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
                P[ch * rows * cols + r * cols + c] = (float)(pixel[ch]);
            }
        }
    }

    result.memory = P;
    result.dimensions[0] = 1;        
    result.dimensions[1] = channels;
    result.dimensions[2] = rows;
    result.dimensions[3] = cols;

    return result;
}

cv::Mat p2i(Ipointer X)
{
    int rows = X.dimensions[2];
    int cols = X.dimensions[3];
    int channels = X.dimensions[1];
    cv::Mat img(rows, cols, CV_8UC(channels));

    for (int ch = 0; ch < channels; ch++) 
    {
        for (int r = 0; r < rows; r++) 
        {
            for (int c = 0; c < cols; c++) 
            {
                img.at<cv::Vec3b>(r, c)[ch] = (uchar)X.memory[ch * img.rows * img.cols + r* img.cols + c];
            }
        }
    }
    
    return img;

}

graph i2n(std::string filepath, int row_size, int col_size)
{

    cv::Mat img = cv::imread(filepath, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::cout << "Failed to load image!\n";
        std::exit(1);
    }
    Ipointer X = i2p(filepath, row_size, col_size);
    auto node = std::make_shared<NodeBackProp>(filepath,X.dimensions[0],X.dimensions[1],X.dimensions[2], X.dimensions[3], 1);
    cudaMemcpy(node->output, X.memory, node->total * sizeof(float), cudaMemcpyHostToDevice);
    free(X.memory);
    return node;
}

cv::Mat n2i(const graph X)
{
    int rows = X->dim[2];
    int cols = X->dim[3];
    int channels = X->dim[1];
    float *cpu_img = (float*)malloc(X->total*sizeof(float));
    cudaMemcpy(cpu_img, X->output, X->total*sizeof(float), cudaMemcpyDeviceToHost);
    cv::Mat img(rows, cols, CV_8UC(channels));
    for (int ch = 0; ch < channels; ++ch) 
    {
        for (int r = 0; r < rows; ++r) 
        {
            for (int c = 0; c < cols; ++c) 
            {   
                uchar val = cpu_img[ch * img.rows * img.cols + r* img.cols + c];
                img.at<cv::Vec3b>(r, c)[ch] = val;
            }
        }
    }
    free(cpu_img);
    return img;

}

Ipointer Bi2p(const str& folder, const int num_images, const int row_size, const int col_size) {
    Text filepaths = ImagePaths(folder, num_images);

    if (filepaths.size() < num_images) {
        std::cerr << "Not enough images in folder\n";
        std::exit(1);
    }

    Ipointer result;
    result.dimensions[0] = num_images;
    result.dimensions[1] = 3; 
    result.dimensions[2] = row_size;
    result.dimensions[3] = col_size;
    result.labels = filepaths;

    float* P = (float*)malloc(3 * num_images * row_size * col_size * sizeof(float));
    if (!P) {
        std::cerr << "Memory allocation failed\n";
        std::exit(1);
    }

    // For each image
    for (int i = 0; i < num_images; i++) {
        cv::Mat img = cv::imread(folder+"/"+filepaths[i], cv::IMREAD_COLOR);
        if (img.empty()) 
        {
            std::cout << "Failed to load image: " << filepaths[i] << "\n";
            free(P);
            std::exit(1);

        }

        cv::resize(img, img, cv::Size(col_size, row_size), 0, 0, cv::INTER_AREA);

        int channels = img.channels();  
        int rows = img.rows;
        int cols = img.cols;
        if(channels != 3)
        {
            std::cout << "Image at path: " << filepaths[i] << " does not have 3 channels \n";
            std::exit(1);
        }

        // Fill in channel-first format (C × H × W)
        for (int ch = 0; ch < channels; ch++) {
        for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) 
        {
                    cv::Vec3b pixel = img.at<cv::Vec3b>(r, c);
                    size_t index = i*channels*rows*cols + ch*rows*cols + r*cols + c;
                    P[index] = (float)pixel[ch];
                }
            }
        }
    }
    result.memory = P;

    return result;
}

graph Bi2n(str filepath,const int num_images, const int row_size , const int col_size)
{

    Ipointer X = Bi2p(filepath, num_images, row_size, col_size);
    std::cout << "Loaded " << num_images << " images from " << X.labels.size() << "\n";
    for(auto & name : X.labels)
    {
        std::cout << name << "\n";
    }
    auto node = std::make_shared<NodeBackProp>(filepath,X.dimensions[0],X.dimensions[1],X.dimensions[2], X.dimensions[3], 1);
    cudaMemcpy(node->output, X.memory, node->total * sizeof(float), cudaMemcpyHostToDevice);
    free(X.memory);
    return node;
}

cv::Mat Bn2i(const graph X)
{
    int batch    = X->dim[0];  
    int channels = X->dim[1];  
    int rows     = X->dim[2];
    int cols     = X->dim[3];

    float* cpu_img = (float*)malloc(X->total * sizeof(float));
    cudaMemcpy(cpu_img, X->output, X->total * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat img(rows, cols * batch, CV_8UC(channels), cv::Scalar(0));

    for (int b = 0; b < batch; ++b) {
    for (int ch = 0; ch < channels; ++ch) {
    for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
        int idx = b * (channels * rows * cols) + ch * (rows * cols) + r * cols + c;
        uchar val = static_cast<uchar>(cpu_img[idx]);
        img.at<cv::Vec3b>(r, b * cols + c)[ch] = val;
    }}}}

    free(cpu_img);
    return img;
}

AdamParameter::AdamParameter(str n, int batch, int out, int in, int row, int col, double norm) : NodeBackProp(n, out, in, row, col,1), lr(LEARNING_RATE), t(1), b1(0.9), b2(0.999), epsilon(1e-8), batch_size(batch), group_norm(norm), weight_decay(0.01f)
{
        std::random_device rd;
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
        total_size = out*in*row*col;
        SafeCudaMalloc("M-matrix of AdamParameter",m,total_size);
        SafeCudaMalloc("V-matrix of AdamParameter",v,total_size);
        const int tpb = THREADSPERBLOCK; 
        const int bpg = (total_size+tpb-1) / tpb;
        Standard_Weights<<<bpg,tpb>>>(output, total_size, sqrtf(XAVIER/(in*row*col)), seed); 
        fillKernel<<<bpg,tpb>>>(m,0.0f,total_size);
        fillKernel<<<bpg,tpb>>>(v,0.0f,total_size);
};

void AdamParameter::save(std::ofstream& f) const 
{
    size_t name_len = op_name.size();
    f.write(reinterpret_cast<const char*>(&name_len), sizeof(size_t));
    f.write(op_name.data(), name_len);
    f.write(reinterpret_cast<const char*>(&total), sizeof(int));

    // Copy device -> host -> file
    std::vector<float> host(total);
    cudaMemcpy(host.data(), output, total * sizeof(float), cudaMemcpyDeviceToHost);
    f.write(reinterpret_cast<const char*>(host.data()), total * sizeof(float));

    // Optionally save Adam moments for resuming training
    cudaMemcpy(host.data(), m, total * sizeof(float), cudaMemcpyDeviceToHost);
    f.write(reinterpret_cast<const char*>(host.data()), total * sizeof(float));
    cudaMemcpy(host.data(), v, total * sizeof(float), cudaMemcpyDeviceToHost);
    f.write(reinterpret_cast<const char*>(host.data()), total * sizeof(float));
}

void AdamParameter::load(std::ifstream& f) 
{
    size_t name_len;
    f.read(reinterpret_cast<char*>(&name_len), sizeof(size_t));
    str loaded_name(name_len, '\0');
    f.read(const_cast<char*>(loaded_name.data()), name_len);

    int loaded_total;
    f.read(reinterpret_cast<char*>(&loaded_total), sizeof(int));

    if (loaded_total != total) 
    {
        std::cerr << "Loaded parameter " << loaded_name << " does not match expected size\n";
        std::exit(1);
    }
    
    std::vector<float> host(total);
    f.read(reinterpret_cast<char*>(host.data()), total * sizeof(float));
    cudaMemcpy(output, host.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    f.read(reinterpret_cast<char*>(host.data()), total * sizeof(float));
    cudaMemcpy(m, host.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    f.read(reinterpret_cast<char*>(host.data()), total * sizeof(float));
    cudaMemcpy(v, host.data(), total * sizeof(float), cudaMemcpyHostToDevice);
    }

void AdamParameter::update(const bool W)
{   
        const int tpb = THREADSPERBLOCK;
        const int bpg = (total_size + tpb - 1) / tpb;
        isNan("Gradient of " + op_name, grad, total_size);
        if(W) AdamWUpdate<<<bpg, tpb>>>(output, grad, total_size, t, m, v, b1, b2,epsilon,weight_decay,lr);
        else  AdamUpdate<<<bpg, tpb>>>(output, grad, total_size,t, m, v, b1, b2, epsilon, lr);
        CheckError("AdamUpdate in AdamParameter update");
        t++;
};

void AdamParameter::accumulate_grad(double* global_scale)
{
        const int tpb = THREADSPERBLOCK;
        const int bpg = (total_size + tpb - 1) / tpb;
        SumSquared<<<bpg, tpb>>>(global_scale, grad, total_size);
        CheckError("SSWarp in Gradient Norm of " + op_name); 
};

void AdamParameter::gradnorm(const double* global_scale){ScalePtr(grad, global_scale, total_size, 1);};

void Dimension(graph X)
{   std::cout<< "Dimensions for node: " << X->op_name << "\n";
    std::cout << "(";
    for(int i=0;i<4; ++i)
    {
        std::cout << " x " << X->dim[i]<<"\t";
    }
    std::cout<< ") \n";
}

void Dimension(AdamParameter* X)
{   std::cout<< "Dimensions for node: " << X->op_name << "\n";
    std::cout << "(";
    for(int i=0;i<4; ++i)
    {
        std::cout << " x " << X->dim[i]<<"\t";
    }
    std::cout<< ") \n";
}

void Zerograd(AdamParameter* X)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (tpb + X->total_size-1) / tpb;
    fillKernel<<<bpg,tpb>>>(X->grad, 0.0f, X->total_size);

}

void Zerograd(graph X)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (tpb + X->total-1) / tpb;
    fillKernel<<<bpg,tpb>>>(X->grad, 0.0f, X->total);

}

void isNan(graph X, const int type )
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb-1)/tpb;
    if (type == 0) ISNAN<<<bpg,tpb>>>(X->output, X->total);
    else ISNAN<<<bpg,tpb>>>(X->grad, X->total);

    CheckError(X->op_name + " has nan value");
    return;
}

void isNan(AdamParameter* X, const int type )
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb-1)/tpb;
    if (type == 0) ISNAN<<<bpg,tpb>>>(X->output, X->total);
    else ISNAN<<<bpg,tpb>>>(X->grad, X->total);

    CheckError(X->op_name + " has nan value");
    return;
}

void printGPU(const graph X, const int type)
{
    int batch = X->dim[0];
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));

    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);
    
    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for(int b = 0; b < batch; ++b)
    {
    std::cout << "_____________________________________ \n";
    std::cout << " -----BATCH " << b << "-----  \n";

        for (int c = 0; c < ch; ++c){
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < rows; ++r){
        for (int col = 0; col < cols; ++col){
        int idx = (b*ch*rows*cols) + (c * rows * cols) + (r * cols) + col;
        std::cout << CPU[idx] << "\t";
        }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }
    std::cout << "_____________________________________ \n \n \n";
    }

    free(CPU);
}

void printGPU(AdamParameter* X, const int type)
{
     int batch = X->dim[0];
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));

    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    
    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for(int b = 0; b < batch; ++b)
    {
    std::cout << "_____________________________________ \n";
    std::cout << " -----BATCH " << b << "-----  \n";

        for (int c = 0; c < ch; ++c){
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < rows; ++r){
        for (int col = 0; col < cols; ++col){
        int idx = (b*ch*rows*cols) + (c * rows * cols) + (r * cols) + col;
        std::cout << CPU[idx] << "\t";
        }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }
    std::cout << "_____________________________________ \n \n \n";
    }

    free(CPU);
}

void printHeadGPU(const graph X, const int type)
{
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));
    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for (int c = 0; c < min(3,ch); ++c)
    {
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < min(5,rows); ++r)
        {
            for (int col = 0; col < min(5,cols); ++col)
            {
                int idx = (c * rows * cols) + (r * cols) + col;
                std::cout << CPU[idx] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }

    free(CPU);
}

void printHeadGPU(AdamParameter* X, const int type)
{
    int ch     = X->dim[1];
    int rows   = X->dim[2];
    int cols   = X->dim[3];
    int total  = X->total;

    float *CPU = (float *)malloc(total * sizeof(float));
    if (type == 0) cudaMemcpy(CPU, X->output, total * sizeof(float), cudaMemcpyDeviceToHost);
    else cudaMemcpy(CPU, X->grad, total * sizeof(float), cudaMemcpyDeviceToHost);

    if (type == 0) std::cout << "Printing dimensions for node " <<X->op_name << "->output \n";
    else std::cout << "Printing dimensions for node " <<X->op_name << "->grad \n";
    for (int c = 0; c < min(3,ch); ++c)
    {
        std::cout << "Channel " << c << ":\n";
        for (int r = 0; r < min(5, rows); ++r)
        {
            for (int col = 0; col < min(5, cols); ++col)
            {
                int idx = (c * rows * cols) + (r * cols) + col;
                std::cout << CPU[idx] << "\t";
            }
            std::cout << "\n";
        }
        std::cout << "------------------------------\n";
    }

    free(CPU);
}

void BMinMaxNorm(const graph &X)
{
    /*
    @brief: Performs Batch Min-Max Normalization on input X for printing purposes. 
    This kernel computes the minimum and maximum values for each batch, normalizes
    the data to the range [0, 255], and prepares it for visualization. 
    It is designed to handle 4D tensors with dimensions (batch_size, channels, height, width) 
    and is optimized for GPU execution.
    */

    const int tpb = THREADSPERBLOCK;
    const int batch_size = X->dim[0];
    const int elements_per_batch = X->total / batch_size;
    
    float* G_max;
    float* G_min;
    SafeCudaMalloc("GPU batch maximum", G_max, batch_size);
    SafeCudaMalloc("GPU batch minimum", G_min, batch_size);

    fillKernel<<<(batch_size + tpb - 1) / tpb, tpb>>>(G_max, -FLT_MAX, batch_size);
    fillKernel<<<(batch_size + tpb - 1) / tpb, tpb>>>(G_min, FLT_MAX, batch_size);

    const int bpg_minmax = (batch_size * X->dim[1] + tpb - 1) / tpb;
    BMax<<<bpg_minmax, tpb>>>(X->output, G_max, batch_size, X->dim[1], elements_per_batch / X->dim[1]);
    BMin<<<bpg_minmax, tpb>>>(X->output, G_min, batch_size, X->dim[1], elements_per_batch / X->dim[1]);
    
    const int bpg_norm = (X->total + tpb - 1) / tpb;
    BatchMinMaxNorm<<<bpg_norm, tpb>>>(X->output, G_max, G_min, batch_size, X->total);
    BatchMinMaxDeNorm<<<bpg_norm, tpb>>>(X->output, 255.0f, 0.0f, batch_size, X->total);
    
    cudaFree(G_max);
    cudaFree(G_min);
    CheckError("BatchMinMaxNorm Kernel");
}

void BMaxNorm(const graph & X)
{
    float *max, *min;
    const int tpb = THREADSPERBLOCK;
    SafeCudaMalloc("Max", max, X->dim[0]);
    SafeCudaMalloc("Min", min, X->dim[0]);
    fillKernel<<<(X->dim[0]+tpb-1)/tpb,tpb>>>(max,255.0f,X->dim[0]);
    fillKernel<<<(X->dim[0]+tpb-1)/tpb,tpb>>>(min,  0.0f,X->dim[0]);
    BatchMinMaxNorm<<<(X->total+tpb-1)/tpb,tpb>>>(X->output, max,min,X->dim[0], X->total); 
    cudaFree(max);
    cudaFree(min);

}

void StandardNorm(const graph &X, const float img_max, const float mean, const float std)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb - 1) / tpb;
    StdNorm<<<bpg, tpb>>>(X->output, img_max, mean, std, X->total);
    CheckError("StandardNorm kernel");
}

void StandardDeNorm(const graph &X, const float img_max, const float mean, const float std)
{
    const int tpb = THREADSPERBLOCK;
    const int bpg = (X->total + tpb - 1) / tpb;
    StdDeNorm<<<bpg, tpb>>>(X->output, img_max, mean, std, X->total);
    CheckError("StandardDeNorm kernel");
}

void BPrintImage(const graph &X, const int row_size, const int col_size)
{
    auto X_in = std::make_shared<NodeBackProp>(X->op_name, X->dim[0],X->dim[1],X->dim[2],X->dim[3],1);
    cudaMemcpy(X_in->output, X->output, X->total*sizeof(float), cudaMemcpyDeviceToDevice);
    CheckError("Memcpy");
    BMinMaxNorm(X_in);
    CheckError("BatchMinMaxNorm in BPrintImage");
    auto image = Bn2i(X_in);

    if(row_size > 0 && col_size > 0) cv::resize(image, image, cv::Size(row_size*X->dim[0], col_size), 0, 0, cv::INTER_AREA);
    
    cv::imshow(X_in->op_name, image);
    cv::waitKey(0);
    X_in->clear();

}

void prepare(const graph &base, const graph &input, const graph &target, int t, int T, const uint64_t seed)
{
    if (input->total != target->total)
    {
        std::cout << "SHAPE MISMATCH IN PREPARATION \n";
        Dimension(input); Dimension(target);
        exit(1);
    }
    const int tpb = THREADSPERBLOCK;
    const int bpg = (input->total+tpb-1)/tpb;

    cudaMemcpy(input->output, base->output, input->total*sizeof(float), cudaMemcpyDeviceToDevice);
    CheckError("CudaMemcpy");

    GaussianNoise<<<bpg,tpb>>>(target->output, target->total, seed);
    CheckError("Gaussian Noise in preparation");

    AddNoise<<<bpg, tpb>>>(input->output, target->output,t, T, input->total);
    CheckError("Addition of Noise in preparation");
}

int ArgMaxToCPU(const graph& input, int* X)
{
    for(int i = 0; i < 3; i++)
    {if (input->dim[i] != 1)
        {
            printf("Dimension does not match kernel"); 
            Dimension(input);
            std::exit(1);
        }
    }
    ArgMax<<<1,1>>>(input->output, X, input->total);
    int max_id;
    cudaMemcpy(&max_id, X, sizeof(int), cudaMemcpyDeviceToHost);
    return max_id;
}

int TopKSampleToCPU(const graph& input, int* X, const int k)
{
    for (int i = 0; i < 3; ++i){
    if (input->dim[i] != 1)
    {
        printf("TopKSample: dimension mismatch at dim[%d]\n", i);
        Dimension(input);
        std::exit(1);
    }}

    if (k > THREADSPERBLOCK || k < 1)
    {
        printf("TopKSample: k must be in [1, 256], got %d\n", k);
        std::exit(1);
    }
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<float> dist(1e-7f, 1.0f - 1e-7f);
    const float rand_val = dist(gen);
    TopKSampleKernel<<<1, 1>>>(input->output, X, input->total, k, rand_val);
    int sampled_id;
    cudaMemcpy(&sampled_id, X, sizeof(int), cudaMemcpyDeviceToHost);
    return sampled_id;
}

graph_tree topological_sort(const graph& root) 
{
    std::unordered_set<NodeBackProp*> visited;
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

str TextualEmbedding::preprocessWord(const str& word)
{
        str word_lower = word;
        size_t dot_pos = word.find_last_of('.');
        if (dot_pos != str::npos && dot_pos == word.length() - 1) {word_lower = word.substr(0, dot_pos);}
        std::transform(word_lower.begin(), word_lower.end(),word_lower.begin(), ::tolower);
        
        return word_lower;
};

TextualEmbedding::TextualEmbedding(const int embed_dim, const int batch_size, const int max_c_len, const int max_vocab_size)
    : MAX_BATCH_SIZE(batch_size), MAX_CONTEXT_LEN(max_c_len), MAX_VOCAB_SIZE(max_vocab_size), embed_dim(embed_dim)
{
        std::random_device rd;
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();

        SafeCudaMalloc("EmbedSpace", EmbedSpace, MAX_VOCAB_SIZE * embed_dim);
        SafeCudaMalloc("Keys", keys, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);

        const int embed_total = MAX_VOCAB_SIZE * embed_dim;
        GaussianNoise<<<(embed_total+tpb-1)/tpb, tpb>>>(EmbedSpace, embed_total, seed);
        
        fillKernel<<<(MAX_BATCH_SIZE*MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(keys, INT_MIN, MAX_BATCH_SIZE * MAX_CONTEXT_LEN);
    
        CheckError("TextualEmbedding initialization");
};

TextualEmbedding::~TextualEmbedding(){cudaFree(EmbedSpace);cudaFree(keys);}

void TextualEmbedding::updateVocabulary(const Text& texts)
{       
        /*
        @author Updates the vocabulary with new words from the input texts. 
        If new words are added, their embeddings are initialized with Gaussian noise.
        */
        std::random_device rd;
        const uint64_t seed =  ((uint64_t)rd() << 32) | rd();

        Text tokens = {"<START>", "<END>", "<UNK>", "<PAD>"}; 
        Text new_texts = tokens; // Ensure special tokens are included
        new_texts.insert(new_texts.end(), texts.begin(), texts.end());
        std::vector<int> new_indices;
        for(const auto& word : new_texts) {
        str word_lower = preprocessWord(word);
        if (Vocabulary.find(word_lower) == Vocabulary.end()) {
        int index = WordSpace.size();
        if(index >= MAX_VOCAB_SIZE - 4) {
            std::cout << "Warning: Vocabulary size exceeded maximum limit of " 
            << MAX_VOCAB_SIZE << ". Skipping word: " << word_lower << "\n";
            continue;
        }
        WordSpace[word_lower] = index;
        KeySpace[index] = word_lower;
        Vocabulary.insert(word_lower);
        new_indices.push_back(index);
        }}
        
        if(!new_indices.empty()) {
        for(int idx : new_indices) {
        const int offset = idx * embed_dim;
        GaussianNoise<<<(embed_dim+tpb-1)/tpb, tpb>>>(EmbedSpace + offset, embed_dim, seed);
        }
        CheckError("Vocabulary update - new embeddings");
        }
};
    
void TextualEmbedding::encodeText(const Text& texts, const int batch_idx)
{
    /*
    @author Encodes a single text input into its corresponding indices in the embedding space.
    The encoded indices are stored in the keys tensor at the position corresponding to the batch index.
    */
    if(batch_idx >= MAX_BATCH_SIZE) 
    {std::cout << "Error: batch_idx " << batch_idx << " exceeds MAX_BATCH_SIZE " << MAX_BATCH_SIZE << "\n";return;}
    
    const int offset = batch_idx * MAX_CONTEXT_LEN;
    fillKernel<<<(MAX_CONTEXT_LEN+tpb-1)/tpb, tpb>>>(keys + offset, INT_MIN, MAX_CONTEXT_LEN);
    const int num_tokens = std::min((int)texts.size(), MAX_CONTEXT_LEN);
    
    for(int i = 0; i < num_tokens; ++i)
    {
    str word_lower = preprocessWord(texts[i]);
    int index = -1;
    auto it = WordSpace.find(word_lower);
    if(it != WordSpace.end()) {index = it->second;} 
    else {std::cout << "Warning: Word '" << word_lower << "' not in vocabulary. Using masked value.\n";}
    if(index >= MAX_VOCAB_SIZE || index < 0) {
    std::cout << "Error: Invalid index " << index << " for word: " << word_lower << "\n";continue;}
    cudaMemcpy(&keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    CheckError("Text encoding");
    }

void TextualEmbedding::rencodeText(const Text& texts, const int batch_idx, const int start_idx)
{
    /*
    @author Recursively encodes a single text input into its corresponding indices in the embedding space
    starting from the given start index. The encoded indices are stored in the keys tensor at the position 
    corresponding to the batch index.
    */
    if(batch_idx >= MAX_BATCH_SIZE) 
    {
        std::cout << "Error: batch_idx " << batch_idx << " exceeds MAX_BATCH_SIZE " << MAX_BATCH_SIZE << "\n";
        std::exit(1);
    }
    
    const int offset = batch_idx * MAX_CONTEXT_LEN + start_idx;
    const int num_tokens = std::min((int)texts.size(), MAX_CONTEXT_LEN - start_idx);
    
    for(int i = 0; i < num_tokens; ++i)
    {
    str word_lower = preprocessWord(texts[i]);
    int index = -1;
    auto it = WordSpace.find(word_lower);
    if(it != WordSpace.end()) {index = it->second;} 
    else {std::cout << "Warning: Word '" << word_lower << "' not in vocabulary. Using masked value.\n";}
    if(index >= MAX_VOCAB_SIZE || index < 0) {
    std::cout << "Error: Invalid index " << index << " for word: " << word_lower << "\n";continue;}
    cudaMemcpy(&keys[offset + i], &index, sizeof(int), cudaMemcpyHostToDevice);
    }
    
    CheckError("Text encoding");
    }

void TextualEmbedding::encodeBatch(const BatchText& batch_texts)
{
    /* @author 
    Encodes batches of texts into their corresponding indices in the embedding space. 
    The encoded indices are stored in the keys tensor at the position corresponding to the batch index.
    */
    const int batch_size = std::min((int)batch_texts.size(), MAX_BATCH_SIZE);
    for(int b = 0; b < batch_size; ++b) 
    {
        updateVocabulary(batch_texts[b]);
        encodeText(batch_texts[b], b);
    
    }}

void TextualEmbedding::forward(const graph&X)
{
        /*
        @author Performs the forward pass to retrieve embeddings for the encoded texts.
        It gathers embeddings from the embedding space based on the keys tensor and returns the output tensor matrices.
        */

        if(X->dim[0] > MAX_BATCH_SIZE || X->dim[2] > MAX_CONTEXT_LEN) 
        {
            printf("Dimension Mismatch... Received (Batch x Context): (%i, %i), Max: (%i, %i)", 
            X->dim[0], X->dim[2], MAX_BATCH_SIZE, MAX_CONTEXT_LEN);std::exit(1);
        }

        const int bpg = (X->total+tpb-1)/tpb;
        GatherEmbeddings<<<bpg, tpb>>>(X->output, EmbedSpace, keys, X->dim[2], MAX_CONTEXT_LEN, embed_dim, X->total);
        CheckError("Forward pass - gather embeddings");
    }
 
void TextualEmbedding::rforward(const graph&X, const int start_idx)
{
        /*
        @author Performs the recursive forward pass to retrieve embeddings for the encoded texts at the bottom of X.
        It gathers embeddings from the embedding space based on the keys tensor and returns the output tensor matrices.
        It also assumes you're only working on batch 1.
        */
        if(X->dim[0] != 1 || X->dim[2] > MAX_CONTEXT_LEN) 
        {
        printf("Dimension Mismatch... Received (Batch x Context): (%i, %i), Max: (%i, %i), can only recursively call on single batches", 
            X->dim[0], X->dim[2], MAX_BATCH_SIZE, MAX_CONTEXT_LEN);std::exit(1);
        }
        
        const int bpg = (embed_dim+tpb-1)/tpb;
        const int xOffset = start_idx * embed_dim;
        GatherEmbeddings<<<bpg, tpb>>>(X->output + xOffset, EmbedSpace, keys + start_idx, 1, 1, embed_dim, embed_dim);
        CheckError("Forward pass - gather embeddings");
    }
        
void TextualEmbedding::one_hot_forward(const graph&X)
{
        /*
        @author Performs a forward pass to retrieve one-hot encoded embeddings for the encoded texts.
        It creates one-hot vectors based on the keys tensor and returns the output tensor matrices.
        */
        if(X->dim[0] > MAX_BATCH_SIZE || X->dim[2] > MAX_CONTEXT_LEN) {
            printf("Dimension Mismatch... Received (Batch x Context): (%i, %i), Max: (%i, %i)", 
            X->dim[0], X->dim[2], MAX_BATCH_SIZE, MAX_CONTEXT_LEN);std::exit(1);
        }
        const int size = X->dim[0] * X->dim[2];
        const int bpg = (X->total+tpb-1)/tpb;
        fillKernel<<<bpg, tpb>>>(X->output, 0.0f, X->total);
        OneHotEmbeddings<<<bpg,tpb>>>(X->output, keys, X->dim[2], MAX_CONTEXT_LEN, Vocabulary.size(), size);
        CheckError("One-hot Forward pass - gather embeddings");
       
}

void TextualEmbedding::EmbeddingUpdate(const graph& X)
{
        /*
        @author Updates the embedding space using gradients from backpropagation. 
        If custom keys are provided, they are used for the update; otherwise, the internal keys are used
        */
       const int tpb = THREADSPERBLOCK; 
       const int bpg = (X->total+tpb-1)/tpb;
       KeyUpdate<<<bpg,tpb>>>(EmbedSpace, X->grad,keys,X->dim[2],MAX_CONTEXT_LEN, embed_dim,LEARNING_RATE,X->total);
       CheckError("Key Update in Embedding Space update in Textual Embedding");
        
}

graph GraphOperations::like(const graph& X, const str name)
{
    /*
    @author Function requires manual clearing of nodes created during graph computation
    */
    auto node = std::make_shared<NodeBackProp>(name, X->dim[0], X->dim[1], X->dim[2], X->dim[3],1);
    node->inputs = {};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    node->forward = [=](){};
    node->backward = [=](){};
    node->zero_grad = [=](){Zerograd(node);};
    node->free = [=](){};
    return node;
}

graph GraphOperations::Permute(const graph& X, const int i0, const int i1, const int i2, const int i3)
{
    const int a = X->dim[0]; const int pa = X->dim[i0];
    const int b = X->dim[1]; const int pb = X->dim[i1];
    const int c = X->dim[2]; const int pc = X->dim[i2];
    const int d = X->dim[3]; const int pd = X->dim[i3];
    int inv_perm[4];
    int perm[4] = {i0, i1, i2, i3};
    for(int i = 0; i < 4; i++) {inv_perm[perm[i]] = i;}
    auto node = std::make_shared<NodeBackProp>("Permuted " + X->op_name, pa, pb, pc, pd, 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1) / tpb;
    node->forward = [=]()
    {
        permute<<<bpg,tpb>>>(X->output, node->output, a, b, c, d, i0, i1, i2, i3);
        CheckError("Permute forward");
    };
    
    node->backward = [=]()
    {
        permute<<<bpg,tpb>>>(node->grad, X->grad, a, b, c, d, inv_perm[0], inv_perm[1], inv_perm[2], inv_perm[3]);
        CheckError("Permute backward");
    };
    
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}

graph GraphOperations::PositionalEncoding(const int &t, const int d_model)
{

    auto node = std::make_shared<NodeBackProp>("SinoSuodalPosEncoding", 1, 1, 1, d_model, 1);
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1) / tpb;
    node->inputs = {};
    node->forward = [=]()
    {
        PEncoding<<<bpg,tpb>>>(node->output, t, d_model, node->total);
        CheckError("SinoSuodal Positional Encoding forward");
    };
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
        
}

graph GraphOperations::MatrixPositionalEncoding(const graph& X, const int start_idx)
{
    if (X->dim[1] != 1)
    {
        std::cerr << "MatrixPositionalEncoding expects input with channel dimension of 1\n";
        Dimension(X);
        std::exit(1);
    }

    const int batch = X->dim[0];
    const int rows = X->dim[2];
    const int cols = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("MatrixSinoSuodalPosEncoding", batch, 1, rows, cols, 1);
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1) / tpb;
    node->inputs = {X};
    node->forward = [=]()
    {
        MatPEncoding<<<bpg,tpb>>>(X->output, node->output,batch,rows,cols, node->total, start_idx);
        CheckError("Matrix SinoSuodal Positional Encoding forward");
    };

    node->backward = [=]()
    {
        Accumulate<<<bpg,tpb>>>(node->grad, X->grad, node->total);
        CheckError("Matrix SinoSuodal Positional Encoding backward");
    };

    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
        
}

graph GraphOperations::Broadcast_Add(const graph& A, const graph& B)
{
    if(A->dim[1] != B->dim[1])
    {
        std::cout << "Channel mismatch in broadcast add \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    const int batch = A->dim[0];
    const int channels = A->dim[1];
    const int a = A->dim[2];
    const int b = A->dim[3];
    const int c = B->dim[2];
    const int d = B->dim[3];

    if(a % c != 0 || b % d != 0)
    {
        std::cout << "Spatial dimensions not divisible in broadcast add \n";
        std::cout << "A spatial: (" << a << ", " << b << "), B spatial: (" << c << ", " << d << ")\n";
        std::cout << "Requires: A_height % B_height == 0 and A_width % B_width == 0\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

    auto node = std::make_shared<NodeBackProp>("Broadcast Add", batch, channels, a, b, 1);
    node->inputs = {A, B};
    const int total_size = batch * channels * a * b;
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    const int B_total = batch * channels * c * d;

    node->forward = [=]()
    {
        isNan(A); isNan(B);
        broadcast_add_general<<<bpg,tpb>>>(A->output, B->output, node->output, batch, channels, a, b, c, d);
        CheckError("Broadcast Add forward");
    };

    node->backward = [=]()
    {
        Accumulate<<<bpg,tpb>>>(node->grad, A->grad, total_size);
        CheckError("Broadcast Add backward - A grad");

        broadcast_add_backward<<<(B_total+tpb-1)/tpb, tpb>>>(node->grad, B->grad, batch, channels, a, b, c, d);
        CheckError("Broadcast Add backward - B grad");
    };

    node->free = [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node;
}

__global__ void isAbove(float* X, const float x, const int total_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total_size) return;
    if(abs(X[idx]) > x)
    { 
        printf("Value of X[idx]: %f \n", X[idx]);
        __trap();
        return;
    }


};

graph GraphOperations::Broadcast_Channel(const graph& A, const graph& B)
{
    if(A->dim[1] != B->dim[3] || B->dim[1] != 1 || B->dim[2] != 1)
    {
        std::cout << "Channel mismatch in broadcast channel \n";
        std::cout << "Expected B shape: [1, 1, 1, channels], got ["<<B->dim[0]<<", "<<B->dim[1]<<", "<<B->dim[2]<<", "<<B->dim[3]<<"]\n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }
    
    const int batch = A->dim[0];
    const int channels = A->dim[1];
    const int a = A->dim[2];
    const int b = A->dim[3];

    auto node = std::make_shared<NodeBackProp>("Broadcast Channel", batch, channels, a, b, 1);
    node->inputs = {A, B};
    const int total_size = batch*channels*a * b;
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;

    node->forward = [=]()
    {
        isNan(A); isNan(B);
        broadcast_add<<<bpg,tpb>>>(A->output, B->output, node->output, batch, channels, a, b);
        CheckError("Broadcast Add forward");
    };

    node->backward = [=]()
    {
        Accumulate<<<bpg,tpb>>>(node->grad, A->grad, total_size);
        CheckError("Broadcast Add backward - A grad");

        Channel_Squeeze1D<<<bpg,tpb>>>(node->grad, B->grad, batch, channels, a, b);
        CheckError("Broadcast Add backward - B grad");
        
    };

    node->free = [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node;
}

graph GraphOperations::Add(const graph& A, const graph& B)
    {
        if(A->dim[0] != B->dim[0] || A->dim[1] != B->dim[1] || A->dim[2] != B->dim[2] || A->dim[3] != B->dim[3])
        {
            std::cout << "Dimension mismatch in Add \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }

        const int batch = A->dim[0];
        const int channels = A->dim[1];
        const int a = A->dim[2];
        const int b = A->dim[3];

        auto node = std::make_shared<NodeBackProp>("Add", batch, channels, a, b, 1);
        node->inputs = {A,B};
        const int total_size = batch*channels*a*b;
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;

        node->forward = [=]()
        {
            isNan(A); isNan(B);
            ScaleAdd<<<bpg,tpb>>>(A->output, B->output, node->output, 1.0, total_size);
            CheckError("Add forward");
        };

        node->backward = [=]()
        {
            Accumulate<<<bpg,tpb>>>(node->grad, A->grad, total_size);
            CheckError("Add backward - A grad");

            Accumulate<<<bpg,tpb>>>(node->grad, B->grad, total_size);
            CheckError("Add backward - B grad");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};

        
        return node;
    }

graph GraphOperations::Last(const graph& X)
{
    if(X->dim[0] != 1 || X->dim[1] != 1)
    {
        std::cout << "Dimension mismatch in Last \n";
        Dimension(X);
        std::exit(1);
    }

    auto node = std::make_shared<NodeBackProp>(X->op_name + " Last", X->dim[0], X->dim[1], 1, X->dim[3], 1);
    node->inputs = {X};
    GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    const int bpg = (node->total+tpb-1)/tpb;
    const int offset = X->total - X->dim[3];
    node->forward = [=]()
    {
        isNan(X);
        cudaMemcpy(node->output,X->output+offset,node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        CheckError("Last forward");
    };

    node->backward = [=](){};
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
    }

graph GraphOperations::MeanSquaredError(const graph& prediction, const graph& target)
{   
        bool val = true;
        for(int i = 0; i < 4; ++i) {if (prediction->dim[i] != target->dim[i]) {val = false;}}
        if(val == false)
        {
            std::cout << "Dimension mismatch \n MSE dimensions are: \n";
            Dimension(prediction);
            Dimension(target);
            std::exit(1);
        }
        
        const int batch = target->dim[0];
        const int channels = target->dim[1];
        const int c = target->dim[2];
        const int d = target->dim[3];
        auto node = std::make_shared<NodeBackProp>("MSE Loss",1,1,1,1,1);
        node->inputs = {prediction, target};
        const int total_size = batch*channels*c*d;
        GB += (double)(prediction->total + target->total + 1) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (prediction->total+tpb - 1) / tpb;

        node->forward = [=]()
        {   
                isNan(prediction); isNan(target);

                Zerograd("Node Output", node->output, 1);
                Zerograd("Node Gradient", node->grad, 1);

                scalarMSE<<<(prediction->total+tpb-1)/tpb, tpb>>>(prediction->output,target->output,node->output,batch,prediction->total);
                ScaleValue(node->output, (float)total_size,1,1);

                isNan(node);
                CheckError("Scalar MSE in MSE forward");
                cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
        };

        node->backward = [=]()
        {   
            deriv_MSE<<<bpg,tpb>>>(prediction->output, target->output, prediction->grad, batch, c, d, total_size); 
            isNan(prediction, 1);
            CheckError("derivative of MSE in MSE backward");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd("Node", node->output, node->total);};
        return node;
    }

graph GraphOperations::CrossEntropy(const graph& prediction, const graph& target)
{   
        bool val = true;
        for(int i = 0; i < 4; ++i) 
        {
            if (prediction->dim[i] != target->dim[i]) {val = false;}
        
        }
        if(val == false)
        {
            std::cout << "Dimension mismatch \n CrossEntropy dimensions are: \n";
            Dimension(prediction);
            Dimension(target);
            std::exit(1);
        }
        
        const int batch = target->dim[0];
        const int channels = target->dim[1];
        const int c = target->dim[2];
        const int d = target->dim[3];
        auto node = std::make_shared<NodeBackProp>("Cross Entropy Loss",1,1,1,1,1);
        node->inputs = {prediction, target};
        const int total_size = batch*channels*c*d;
        GB += (double)(prediction->total + target->total + 1) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (prediction->total+tpb-1)/tpb;

        node->forward = [=]()
        {   
                isNan(prediction); isNan(target);

                Zerograd("Node Output", node->output, 1);
                Zerograd("Node Gradient", node->grad, 1);

                scalarCE<<<(prediction->total+tpb-1)/tpb, tpb>>>(prediction->output,target->output,node->output,batch,prediction->total);
                ScaleValue(node->output, (float)batch,1,1);

                isNan(node);
                CheckError("Scalar CE in CE forward");
                cudaMemcpy(&loss, node->output, sizeof(float), cudaMemcpyDeviceToHost);
        };

        node->backward = [=]()
        {   
            deriv_CE<<<bpg,tpb>>>(prediction->output, target->output, prediction->grad, batch, c, d, total_size); 
            isNan(prediction, 1);
            CheckError("derivative of CE in CE backward");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd("Node", node->output, node->total);};
        return node;
    }

graph GraphOperations::SoftMaxCrossEntropy(const graph& prediction, const graph& target)
{   
        bool val = true;
        for(int i = 0; i < 4; ++i) {if (prediction->dim[i] != target->dim[i]) {val = false;}}
        if(val == false)
        {
            std::cout << "Dimension mismatch \n CrossEntropy dimensions are: \n";
            Dimension(prediction);
            Dimension(target);
            std::exit(1);
        }
        
        const int batch = target->dim[0];
        const int channels = target->dim[1];
        const int c = target->dim[2];
        const int d = target->dim[3];
        auto node = std::make_shared<NodeBackProp>("SoftMaxCrossEntropy Loss",batch,channels,c,d,1);
        node->inputs = {prediction, target};
        const int total_size = batch*channels*c*d;
        float* softmax_arr, *maxArr;
        SafeCudaMalloc("Softmax array", softmax_arr, batch*d);
        SafeCudaMalloc("Max array", maxArr, batch*d);
        GB += (double)(node->total + 1) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (prediction->total+tpb-1) / tpb;
        node->forward = [=]()
        {   
            isNan(prediction); isNan(target);
            SoftMax(prediction->output,softmax_arr, node->grad, maxArr, batch,c,d,0);
            scalarCE<<<(prediction->total+tpb-1)/tpb,tpb>>>(node->grad,target->output,node->output,batch,prediction->total);
            ScaleValue(node->output, (float)batch,1,1);
            isNan(node);
            CheckError("Scalar SCE in SCE forward");
            cudaMemcpy(&loss,node->output,sizeof(float),cudaMemcpyDeviceToHost);
        };

        node->backward = [=]()
        {   
            ScaleAdd<<<bpg,tpb>>>(node->grad, target->output, prediction->grad,-1.0f, total_size);
            ScaleValue(prediction->grad, batch, prediction->total, 1);
            isNan(prediction, 1);
            CheckError("derivative of CE in CE backward");
        };

        node->free = [=]()
        {
            node->clear();
            cudaFree(softmax_arr);
            cudaFree(maxArr);
        };
        
        node->zero_grad = [=](){Zerograd(node);Zerograd("Node",node->output,1);};

        return node;
    }

graph GraphOperations::BMM(const graph& A, const graph& B) // m x n, n x p = m x p
    {
        if(A->dim[0] != B->dim[0] || A->dim[3] != B->dim[2] || A->dim[1] != 1 || B->dim[1] != 1){
        if (A->dim[1] != 1 || B->dim[1] != 1){
            std::cout << "BMM currently only supports A and B with shape [batch, 1, a, b] and [batch, 1, b, c] respectively \n";}
            std::cout << "Dimension mismatch in BMM \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }


        const int batch = A->dim[0];
        const int m = A->dim[2];
        const int n = A->dim[3];
        const int p = B->dim[3];

        auto node = std::make_shared<NodeBackProp>("BMM", batch, 1, m, p, 1);
        node->inputs = {A,B};
        GB += (double)(node->total + A->total + B->total) * sizeof(float) / (pow(2,30));
        const int tpb = THREADSPERBLOCK;
        const int bpg = (node->total+tpb-1)/tpb;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch);
        dim3 grid2((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch);
        
        node->forward = [=]()
        {   
            isNan(A); isNan(B);
            bmm<<<grid, block>>>(A->output, B->output, node->output,batch,m,n,p); //Assignment
            CheckError("BMM... A * B in GraphOperations BMM forward");
        };

        node->backward = [=]()
        {
            bmmABT<<<grid, block>>>(node->grad, B->output, A->grad, batch, m, p, n,1); // ∂A = ∂Z * B^T
            CheckError("BMM.. ∂A = ∂Z * B^T in GraphOperations BMM backward");

            bmmATB<<<grid2, block>>>(A->output, node->grad, B->grad,batch, m, n, p,1); // ∂B = A^T * ∂Z
            CheckError("MatMul... X^T*∂Z in GraphOperations BMM backward");
        };
        
        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
    }

graph GraphOperations::BMMABT(const graph& A, const graph& B) //  m x n, p x n = m x p
    {
        if(A->dim[0] != B->dim[0] || A->dim[3] != B->dim[3] || A->dim[1] != 1 || B->dim[1] != 1){
    if (A->dim[1] != 1 || B->dim[1] != 1){ std::cout << "BMM is only made to support A and B with shape [batch, 1, a, b] and [batch, 1, b, c] respectively \n";}
        std::cout << "Dimension mismatch in BMM \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
    }

        const int batch = A->dim[0];
        const int m = A->dim[2];
        const int n = A->dim[3];
        const int p = B->dim[2];

        auto node = std::make_shared<NodeBackProp>("BMMABT", batch, 1, m, p, 1);
        node->inputs = {A,B};

        GB += (double)(node->total + A->total + B->total) * sizeof(float) / (pow(2,30));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch);  // for m×n output
        dim3 grid_dB((n+BLOCK_SIZE-1)/BLOCK_SIZE,(p+BLOCK_SIZE-1)/BLOCK_SIZE,batch);  // for p×n output

        node->forward = [=]()
        {   
            isNan(A); isNan(B);
            bmmABT<<<grid, block>>>(A->output, B->output, node->output,batch,m,n,p); 
            CheckError("BMM... A * B^T in GraphOperations BMMABT forward");
        };

        node->backward = [=]()
        {
            bmm<<<grid_dA, block>>>(node->grad, B->output, A->grad, batch, m, p, n, 1);
            CheckError("BMM.. ∂A = ∂C × B in GraphOperations BMMABT backward");

            bmmATB<<<grid_dB, block>>>(node->grad, A->output, B->grad, batch, p, m, n, 1);
            CheckError("BMMABT... ∂C^T × A in GraphOperations BMMABT backward");
        };
    
        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
    }

graph GraphOperations::BMMATB(const graph& A, const graph& B) // m x n, m x p = n x p
    {
        if(A->dim[0] != B->dim[0] || A->dim[2] != B->dim[2] || A->dim[1] != 1 || B->dim[1] != 1){
        if (A->dim[1] != 1 || B->dim[1] != 1)
        {std::cout << "BMM is only made to support A and B with shape [batch, 1, a, b] and [batch, 1, b, c] respectively \n";}

        std::cout << "Dimension mismatch in BMM \n";
        Dimension(A);
        Dimension(B);
        std::exit(1);
        }

        const int batch = A->dim[0];
        const int m = A->dim[2];
        const int n = A->dim[3];
        const int p = B->dim[3];


        auto node = std::make_shared<NodeBackProp>("BMMATB", batch, 1, n, p, 1);
        node->inputs = {A,B};
        GB += (double)(node->total + A->total + B->total) * sizeof(float) / (pow(2,30));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch);  // for m×n output
        dim3 grid_dB((p+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch);  // for m×p output

        node->forward = [=]()
        {   
            isNan(A); isNan(B);
            bmmATB<<<grid, block>>>(A->output, B->output, node->output,batch,m,n,p); 
            CheckError("BMM... A^T * B in GraphOperations BMMATB forward");
        };

        node->backward = [=]()
        {

            bmmABT<<<grid_dA, block>>>(B->output, node->grad, A->grad, batch, m, p, n, 1);
            CheckError("BMM.. ∂A = B × ∂C^T in GraphOperations BMMATB backward");

            bmm<<<grid_dB, block>>>(A->output, node->grad, B->grad, batch, m, n, p, 1);
            CheckError("BMM... ∂B = A × ∂C in GraphOperations BMMATB backward");
        };

        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
    }

graph GraphOperations::BMMATBT(const graph& A, const graph& B) // m x n, p x m = n x p
{
        if(A->dim[0] != B->dim[0] || A->dim[2] != B->dim[3] || A->dim[1] != 1 || B->dim[1] != 1){
        if (A->dim[1] != 1 || B->dim[1] != 1){
            std::cout << "BMM is only made to support A and B with shape [batch, 1, a, b] and [batch, 1, b, c] respectively \n";}
            std::cout << "Dimension mismatch in BMM \n";
            Dimension(A);
            Dimension(B);
            std::exit(1);
        }
        
        const int batch = A->dim[0];
        const int m = A->dim[2];
        const int n = A->dim[3];
        const int p = B->dim[2];

        auto node = std::make_shared<NodeBackProp>("BMMATBT", batch, 1, n, p, 1);
        node->inputs = {A,B};
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((p+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE,batch);
        dim3 grid_dA((n+BLOCK_SIZE-1)/BLOCK_SIZE,(m+BLOCK_SIZE-1)/BLOCK_SIZE,batch);  // for m×n output
        dim3 grid_dB((m+BLOCK_SIZE-1)/BLOCK_SIZE,(p+BLOCK_SIZE-1)/BLOCK_SIZE,batch);  // for p×m output

        node->forward = [=]()
        {   
            isNan(A); isNan(B);
            bmmATBT<<<grid, block>>>(A->output, B->output, node->output,batch,m,n,p); 
            CheckError("BMM... A^T * B^T in GraphOperations BMMATBT forward");
        };

        node->backward = [=]()
        {
            bmmATBT<<<grid_dA, block>>>(B->output, node->grad, A->grad, batch, p, m, n, 1);
            CheckError("BMM.. ∂A = B^T × ∂C^T in GraphOperations BMMATBT backward");
        
            bmmATB<<<grid_dB, block>>>(node->grad, A->output, B->grad, batch, n, p, m, 1);
            CheckError("BMMATBT... ∂C^T × A^T in GraphOperations BMMATBT backward");
        };
    
    node->free = [=](){node->clear();};
    node->zero_grad = [=](){Zerograd(node);};
    return node;
}
    
graph GraphOperations::SOFTMAX(const graph& X, const int type) // type 0: row-wise, type 1: column-wise
{
        const int a = X->dim[0]; const int b = X->dim[1];    
        const int c = X->dim[2]; const int d = X->dim[3];
        if (b != 1)
        {
            std::cout << "Softmax currently only supports b=1 \n";
            Dimension(X);
            std::exit(1);
        }

        auto node = std::make_shared<NodeBackProp>("Softmax", a,b,c,d, 1);
        node->inputs = {X};    
        GB += (double)node->total * sizeof(float) / (pow(2,30));
        const int arr_size = (type == 0) ? a*c : a*d;
        const int max_size = (type == 0) ? c : d;
        float *arr, *maxArr;
        SafeCudaMalloc("Softmax array", arr, arr_size);
        SafeCudaMalloc("Max array", maxArr, max_size);

        node->forward = [=]() 
        {   
            isNan(X);
            SoftMax(X->output, arr, node->output, maxArr, a, c, d, type);
            CheckError("Softmax forward");

        };

        node->backward = [=]() 
        {
            deriv_SoftMax(node->output,node->grad,X->grad,a,c,d,type);
            CheckError("Deriv Softmax in Softmax");
            isNan(X, 1);
        };

        node->free =  [=]()
        {
            node->clear();
            cudaFree(arr);
            cudaFree(maxArr);
        };
        
        node->zero_grad = [=](){Zerograd(node);};
        return node;

    }

graph GraphOperations::SOFTMASK(const graph& X, const int type) // type 0: row-wise, type 1: column-wise
{
        const int a = X->dim[0]; const int b = X->dim[1];    
        const int c = X->dim[2]; const int d = X->dim[3];
        if (b != 1)
        {
            std::cout << "Softmask currently only supports b=1 \n";
            Dimension(X);
            std::exit(1);
        }

        auto node = std::make_shared<NodeBackProp>("Softmask", a,b,c,d, 1);
        node->inputs = {X};    
        GB += (double)node->total * sizeof(float) / (pow(2,30));
        const int arr_size = (type == 0) ? a*c : a*d;
        const int max_size = (type == 0) ? c : d;
        float* arr, *maxArr;
        SafeCudaMalloc("Softmask array", arr, arr_size);
        SafeCudaMalloc("Max array", maxArr, max_size);

        node->forward = [=]() 
        {   
            isNan(X);
            SoftMask(X->output, arr, node->output, maxArr, a, c, d, type);
            CheckError("Softmask forward");
        };

        node->backward = [=]() 
        {
            deriv_SoftMax(node->output, node->grad, X->grad,a,c,d,type);
            CheckError("Deriv Softmask in Softmask");
            isNan(X, 1);
        };

        node->free =  [=]()
        {
            node->clear();
            cudaFree(arr);
            cudaFree(maxArr);
        };
        
        node->zero_grad = [=](){Zerograd(node);};
        return node;

    }

graph GraphOperations::Scale(const graph& input, const float scale)
{

    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    auto node = std::make_shared<NodeBackProp>("Scale", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        isNan(input);
        ScaleGraph(input->output,node->output,scale,node->total);
        CheckError("Scale Value in Scale forward");

    };

    node->backward = [=]() 
    {
        ScaleGraph(node->grad, input->grad, scale, node->total,1);
        CheckError("Deriv ReLU in RELU");
        isNan(input, 1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::RELU(const graph& input)
{

    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("ReLU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        isNan(input);
        ReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, a*b*c*d); // Assignment operation
        CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        deriv_ReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        CheckError("Deriv ReLU in RELU");
        isNan(input,1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::SILU(const graph& input)
{

    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("SiLU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        isNan(input);
        SiLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, a*b*c*d); 
        CheckError("RELU in RELU forward");

    };

    node->backward = [=]() 
    {
        deriv_SiLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        CheckError("Deriv ReLU in RELU");
        isNan(input, 1);
    };

    node->free =  [=](){node->clear();};

    node->zero_grad = [=](){Zerograd(node);};

    
    return node; 
    }

graph GraphOperations::LeakyRELU(const graph& input)
    {
    const int a = input->dim[0];
    const int b = input->dim[1];
    const int c = input->dim[2];
    const int d = input->dim[3];
    const int tpb = THREADSPERBLOCK;
    auto node = std::make_shared<NodeBackProp>("LeakyReLU", a,b,c,d, 1);
    node->inputs = {input};    
    GB += (double)node->total * sizeof(float) / (pow(2,30));

    node->forward = [=]() 
    {   
        isNan(input);
        LeakyReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output,node->output, node->total); // Assignment operation
        CheckError("LeakyRELU in LeakyRELU forward");

    };

    node->backward = [=]() 
    {
        
        deriv_LeakyReLU<<<(node->total+tpb-1)/tpb,tpb>>>(input->output, node->grad,input->grad,node->total);
        CheckError("Deriv LeakyReLU in LeakyRELU");
        isNan(input, 1);
    };

    node->free = [=]()
    {
        node->clear(); 
    };

    node->zero_grad = [=](){Zerograd(node);};
    
    
    return node; 
    }

graph GraphOperations::CopyCrop(const graph& input1, const graph& input2) // @Channel wise concatenation with cropping or padding as necessary
{   
        const int batch = input2->dim[0];
        const int depth =  input2->dim[1];
        const int d1  = input1->dim[1];
        const int a1 = input1->dim[2];
        const int b1 = input1->dim[3];
        const int a = input2->dim[2];
        const int b = input2->dim[3];
        const int tpb = THREADSPERBLOCK;
        auto node = std::make_shared<NodeBackProp>("CopyNCrop", batch,input1->dim[1] + depth,a,b,1);
        node->inputs = {input1, input2};
        if (a1 != a || b1 != b)
        {
            float *temp, *tGrad; 
            GB += 3 * (double)(node->total) * sizeof(float) / (pow(2,30));
            SafeCudaMalloc("Temp of CopyCrop", temp, batch * depth * a * b);
            SafeCudaMalloc("TGrad of CopyCrop",tGrad, batch * depth * a * b);
            node->forward = [=]()
            {   
                isNan(input1); isNan(input2);
                CopynCrop<<<(tpb+input1->total-1)/tpb, tpb>>>(input1->output, temp, batch, d1,a1,b1,a,b); //Assignment
                CheckError("CopynCrop in CopynCrop");

                BConcatenate<<<(tpb+node->total-1)/tpb,tpb>>>(temp,input2->output,node->output,batch,d1,depth,a,b); //Assignment
                CheckError("Concatenation in CopynCrop");
                
            };

            node->backward= [=]()
            {
                
                BSplit<<<(tpb+node->total-1)/tpb, tpb>>>(tGrad,input2->grad,node->grad,batch,d1,depth,a,b);
                CheckError("Split in CopyCrop");
                
                PaddingCrop<<<(tpb+input1->total-1)/tpb, tpb>>>(tGrad, input1->grad,batch,d1,a1,b1,a,b);
                CheckError("PaddingCrop in CopynCrop");
                isNan(input1, 1); isNan(input2, 1);
            };  

            node->free = [=]()
            {
                node->clear();
                cudaFree(temp);
                cudaFree(tGrad);
            };

            node->zero_grad = [=]()
            {
                Zerograd(node);
                Zerograd("TGrad", tGrad, batch*depth*a*b);
            };
        }
         
        else
        {
            GB += 2 * (double)(node->total) * sizeof(float) / (pow(2,30));
            node->forward = [=]()
            {
                isNan(input1); isNan(input2);
                BConcatenate<<<(tpb+node->total-1)/tpb,tpb>>>(input1->output,input2->output,node->output,batch,d1,depth,a,b);
                CheckError("Concatenation in CopyCrop");
            };
            
            node->backward= [=]()
            {
                
                BSplit<<<(tpb+node->total-1)/tpb, tpb>>>(input1->grad,input2->grad,node->grad,batch,d1,depth,a,b);
                CheckError("Split in CopyNCrop");
                isNan(input1, 1);isNan(input2, 1);
            };

            node->free = [=](){node->clear();};

            node->zero_grad = [=](){Zerograd(node);};
        }
        
        
        return node;
    }

graph GraphOperations::CopyConcat(const graph& input1, const graph& input2) // @Column wise concatenation without cropping
{   
        const int batch = input2->dim[0];
        const int depth =  input2->dim[1];
        const int d1  = input1->dim[1];
        const int a1 = input1->dim[2];
        const int b1 = input1->dim[3];
        const int a = input2->dim[2];
        const int b = input2->dim[3];
        const int tpb = THREADSPERBLOCK;
        auto node = std::make_shared<NodeBackProp>("CopyConcat", batch,depth,a,b+b1,1);
        node->inputs = {input1, input2};
        if (d1 != depth || a1 != a)
        {
            std::cout << "CopyConcat currently only supports concatenation of tensors with the same channels and rowshape \n";
            Dimension(input1);
            Dimension(input2);
            std::exit(1);
        }
   
        GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        node->forward = [=]()
        {   
            isNan(input1); isNan(input2);
            CConcatenate<<<(tpb+node->total-1)/tpb,tpb>>>(input1->output,input2->output,node->output,batch,d1,a,b1,b);
            CheckError("Concatenation in CopyConcat");
        };

        node->backward= [=]()
        {
            CSplit<<<(tpb+node->total-1)/tpb, tpb>>>(input1->grad,input2->grad,node->grad,batch,d1,a,b1,b);
            CheckError("Split in CopyConcat");
            isNan(input1, 1);isNan(input2, 1);
        };
        return node;
    }

graph GraphOperations::LayerNorm(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    const float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;
    auto node = std::make_shared<NodeBackProp>("LayerNorm",a,b,c,d,1);
    double *mean, *std, *ggamma_mean, *ggammanode_mean;
    float *ggamma, *ggammanode; 
    const int tpb = THREADSPERBLOCK;

    SafeCudaMalloc("LayerNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("LayerNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("LayerNorm mean", mean, a);
    SafeCudaMalloc("LayerNorm  std", std,  a);
    SafeCudaMalloc("LayerNorm ggamma_mean", ggamma_mean, a);
    SafeCudaMalloc("LayerNorm ggammanode_mean", ggammanode_mean,  a);

    const size_t smem = tpb * sizeof(double);
    node->inputs = {X};

    node->forward = [=]()
    {
        isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        CheckError("CudaMemcpy of LayerNorm");

        LayerMean<<<a, tpb, smem>>>(X->output,mean,a,b,c,d); // Assigment
        CheckError("LayerMean of LayerNorm");

        LayerStd<<<a, tpb, smem>>>(X->output,mean,std,a,b,c,d); // Assignment
        CheckError("LayerStd of LayerNorm");

        LNorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,c,d,gamma,beta,epsilon); // Assignment
        CheckError("LNorm of LayerNorm");
    };

    node->backward = [=]()
    {
        isNan(node, 1);
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        CheckError("Memset and Memcpy in LNorm Backward");

        ScaleValue(ggamma,gamma,node->total);
        CheckError("Scale in LNorm Backward");

        Multiply<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma,node->output,ggammanode,node->total);
        CheckError("Multiply");

        LayerMean<<<a, tpb, smem>>>(ggamma, ggamma_mean, a,b,c,d, false);
        LayerMean<<<a, tpb, smem>>>(ggammanode, ggammanode_mean, a,b,c,d, false);
        CheckError("LayerMean of ggammas");

        LayerBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);
        CheckError("LayerBackward of LayerNorm");
    };

    node->free = [=]()
        {
            node->clear();
            cudaFree(mean);
            cudaFree(std);
            cudaFree(ggamma);
            cudaFree(ggammanode);
            cudaFree(ggamma_mean);
            cudaFree(ggammanode_mean);
            
        };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd("ggamma", ggamma, node->total);
        Zerograd("ggammanode", ggammanode, node->total);
        Zerograd("ggamma_mean", ggamma_mean, a);
        Zerograd("ggammannode_mean", ggammanode_mean, a);
    };

    return node;
}

graph GraphOperations::BatchNorm(const graph& X)
{
    const int a = X->dim[0], b = X->dim[1], c = X->dim[2], d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("BatchNorm",a,b,c,d,1);
    const float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;
    double *mean, *std, *ggamma_mean, *ggammanode_mean;
    float *ggamma, *ggammanode;
    const int tpb = THREADSPERBLOCK;
    SafeCudaMalloc("BatchNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("BatchNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("BatchNorm mean", mean, b);
    SafeCudaMalloc("BatchNorm  std", std,  b);
    SafeCudaMalloc("BatchNorm ggamma_mean", ggamma_mean, b);
    SafeCudaMalloc("BatchNorm ggammanode_mean", ggammanode_mean,  b);
    node->inputs = {X};

    const size_t smem = tpb * sizeof(double);
        
    node->forward = [=]()
    {
        isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        CheckError("CudaMemcpy of BatchNorm");
            
        BatchMean<<<b,tpb, smem>>>(X->output,mean,a,b,c,d); // Assignment
        CheckError("BatchMean in BatchNorm");

        BatchStd<<<b, tpb, smem>>>(X->output,mean,std,a,b,c,d); // Assignment
        CheckError("Batch Std in BatchNorm");

        BNorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,c,d,gamma,beta, epsilon); // Assignment
        CheckError("BNorm in BatchNorm");        
    };

    node->backward = [=]()
    {
        isNan(node,1);
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        CheckError("Memset and Memcpy in BNorm Backward");

        ScaleValue(ggamma,gamma, node->total);
        CheckError("Scale in BNorm Backward");

        Multiply<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma, node->output, ggammanode, node->total);
        CheckError("Multiply");

        BatchMean<<<b,tpb, smem>>>(ggamma, ggamma_mean, a,b,c,d, false);
        BatchMean<<<b,tpb, smem>>>(ggammanode, ggammanode_mean, a,b,c,d, false);
        CheckError("BatchMean of ggammas");
        BatchBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);

    };
        
    node->free = [=]()
    {
        node->clear();
        cudaFree(mean);
        cudaFree(std);
        cudaFree(ggamma_mean);
        cudaFree(ggammanode_mean);
        cudaFree(ggamma);
        cudaFree(ggammanode);
    };
        
    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd("ggamma", ggamma, node->total);
        Zerograd("ggammanode", ggammanode, node->total);
        Zerograd("ggamma_mean", ggamma_mean, b);
        Zerograd("ggammannode_mean", ggammanode_mean, b);
    };

    return node;
}

graph GraphOperations::GroupNorm(const graph& X, const int group)
{
    const int a = X->dim[0],b = X->dim[1],c = X->dim[2],d = X->dim[3];
        
    if(b%group != 0)
    {
        std::cout << "Groups cannot be cleanly split \n";
        Dimension(X);std::exit(1);
    }
        
    auto node = std::make_shared<NodeBackProp>("GroupNorm",a,b,c,d,1);
    const float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;
    double *mean, *std, *ggamma_mean, *ggammanode_mean;
    float *ggamma, *ggammanode;
    const int tpb = THREADSPERBLOCK;

    SafeCudaMalloc("GroupNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("GroupNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("GroupNorm mean", mean, a*group);
    SafeCudaMalloc("GroupNorm  std", std,  a*group);
    SafeCudaMalloc("GroupNorm ggamma_mean", ggamma_mean, a*group);
    SafeCudaMalloc("GroupNorm ggammanode_mean", ggammanode_mean,  a*group);
    
    node->inputs = {X};
    const size_t smem = tpb * sizeof(double);
        
    node->forward = [=]()
    {
        isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        CheckError("CudaMemcpy of GroupNorm");

        GroupMean<<<a*group,tpb, smem>>>(X->output,mean,a,b,group,c,d);
        CheckError("GroupMean of GroupNorm");

        GroupStd<<<a*group,tpb, smem>>>(X->output,mean,std,a,b,group,c,d); 
        CheckError("GroupStd of GroupNorm");

        GNorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,group,c,d,gamma, beta, epsilon); // Assignment
        CheckError("GNorm of GroupNorm");
    };

    node->backward = [=]()
    {   

        isNan(node, 1);
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        CheckError("Memset and Memcpy in GroupNorm Backward");

        ScaleValue(ggamma,gamma, node->total);
        CheckError("Scale in GroupNorm Backward");

        Multiply<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma, node->output, ggammanode, node->total);
        CheckError("Multiply");

        GroupMean<<<a*group,tpb, smem>>>(ggamma, ggamma_mean, a,b,group,c,d,false);
        GroupMean<<<a*group,tpb, smem>>>(ggammanode, ggammanode_mean,a,b,group,c,d,false);

        CheckError("GroupMean of ggammas");
        GroupBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,group,c,d);
    };
        
    node->free = [=]()
    {
        node->clear(); cudaFree(mean); cudaFree(std); cudaFree(ggamma_mean);
        cudaFree(ggammanode_mean); cudaFree(ggamma); cudaFree(ggammanode);
    };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd("ggamma", ggamma, node->total);
        Zerograd("ggammanode", ggammanode, node->total);
        Zerograd("ggamma_mean", ggamma_mean, a*group);
        Zerograd("ggammannode_mean", ggammanode_mean, a*group);
    };

    return node;

}

graph GraphOperations::InstanceNorm(const graph & X)  
{
    const int a = X->dim[0];
    const int b = X->dim[1];
    const int c = X->dim[2];
    const int d = X->dim[3];
    auto node = std::make_shared<NodeBackProp>("InstanceNorm",a,b,c,d,1);
    const float gamma = 1.0f;
    const float beta = 0.0f;
    const float epsilon = 1e-5f;
    double *mean, *std,*ggamma_mean, *ggammanode_mean;;
    float *ggamma, *ggammanode;
    const int tpb = THREADSPERBLOCK;

    SafeCudaMalloc("InstanceNorm ggamma", ggamma, node->total);
    SafeCudaMalloc("InstanceNorm ggammanode", ggammanode, node->total);
    SafeCudaMalloc("InstanceNorm mean", mean, a*b);
    SafeCudaMalloc("InstanceNorm  std", std,  a*b);
    SafeCudaMalloc("InstanceNorm ggamma_mean", ggamma_mean, a*b);
    SafeCudaMalloc("InstanceNorm ggammanode_mean", ggammanode_mean,  a*b);
    node->inputs = {X};
    const size_t smem = tpb * sizeof(double);
        
    node->forward = [=]()
    {
        isNan(X);
        cudaMemcpy(node->output, X->output, node->total*sizeof(float),cudaMemcpyDeviceToDevice);
        CheckError("CudaMemcpy of InstanceNorm");

        InstanceMean<<<a*b,tpb,smem>>>(X->output,mean,a,b,c,d);
        CheckError("InstanceMean of InstanceNorm");

        InstanceStd<<<a*b,tpb,smem>>>(X->output,mean,std,a,b,c,d); 
        CheckError("InstanceStd of InstanceNorm");

        INorm<<<(node->total+tpb-1)/tpb,tpb>>>(node->output,mean,std,a,b,c,d,gamma, beta, epsilon); //Assignment
        CheckError("INorm of InstanceNorm");
    };

    node->backward = [=]()
    {
        isNan(node,1);
            
        cudaMemcpy(ggamma, node->grad, node->total*sizeof(float), cudaMemcpyDeviceToDevice);
        CheckError("Memset and Memcpy in GroupNorm Backward");

        ScaleValue(ggamma,gamma,node->total);
        CheckError("Scale in GroupNorm Backward");

        Multiply<<<(node->total+tpb-1)/tpb,tpb>>>(ggamma, node->output, ggammanode, node->total);
        CheckError("Multiply");

        InstanceMean<<<a*b,tpb, smem>>>(ggamma, ggamma_mean, a,b,c,d, false);
        InstanceMean<<<a*b,tpb, smem>>>(ggammanode, ggammanode_mean,a,b,c,d, false);

        CheckError("GroupMean of ggammas");
        InstanceBackward<<<(node->total+tpb-1)/tpb,tpb>>>(X->grad,node->output,node->grad,ggamma_mean,ggammanode_mean,std,gamma,epsilon,a,b,c,d);

    };
        
    node->free = [=]()
    {
        node->clear(); cudaFree(mean); cudaFree(std); cudaFree(ggamma_mean); cudaFree(ggammanode_mean); cudaFree(ggamma); cudaFree(ggammanode);
    };

    node->zero_grad = [=]()
    {
        Zerograd(node); Zerograd("ggamma", ggamma, node->total); Zerograd("ggammanode", ggammanode, node->total);
        Zerograd("ggamma_mean", ggamma_mean, a*b); Zerograd("ggammannode_mean", ggammanode_mean, a*b);
    };
        
    return node;

}
  
void GraphOperations::clipNorm(double* global_scale) {for(auto&node : nodes) if(node->clipnorm) node->clipnorm(global_scale);}
    
void GraphOperations::accumulate(double* global_scale) 
{
        for(auto&node : nodes) if(node->accumulate) node->accumulate(global_scale);
        Sqrt_Scale<<<1,1>>>(global_scale,1.0f,0);
}

void GraphOperations::ParameterUpdate() {for(auto&node : nodes) if(node->updateParams) node->updateParams();}

void GraphOperations::forward() 
{   
    for (auto& node : nodes) {if (node->forward) {node->forward();}}
}

void GraphOperations::backward() 
{
for(auto it=nodes.rbegin();it!=nodes.rend();++it){if((*it)->backward)(*it)->backward();} 
}

void GraphOperations::zero_grad() 
{
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it){   
    if ((*it)->zero_grad)
    {
        (*it)->zero_grad();
    }}
}

void GraphOperations::printNodes(const bool display_grad) 
{
    for (auto& node : nodes) {
    if (node->zero_grad) 
    {   
    std::cout << "Calling Node: " << node->op_name << "\n"; if (display_grad) printHeadGPU(node,1);
    }}
}

void GraphOperations::clear_graph(){for (auto &node: nodes){if(node->free){node->free();}}nodes.clear();}

Identity::Identity(GraphOperations& go_ref, const str name) : go(go_ref), name(name) {}
graph Identity::forward(const graph& X) 
{
        auto node = std::make_shared<NodeBackProp>(name,X->dim[0],X->dim[1],X->dim[2],X->dim[3],1);
        go.GB += (double)(node->total) * sizeof(float) / (pow(2,30));
        node->inputs = {X};
        node->forward  = [=](){cudaMemcpy(node->output, X->output, node->total*sizeof(float), cudaMemcpyDeviceToDevice);};
        node->backward = [=](){cudaMemcpy(X->grad,  node->grad, X->total*sizeof(float), cudaMemcpyDeviceToDevice);};
        node->free = [=](){node->clear();};
        node->zero_grad = [=](){Zerograd(node);};
        return node;
}

Linear::Linear(GraphOperations &go_ref, const int input, const int output, const str name) : go(go_ref), in(input), out(output) 
{   
    if (name != "") op_name = name;
    W1 =  new AdamParameter(name + " W1",1,1,1,in,out);
    B1 =  new AdamParameter(name + " B1",1,1,1,1,out);     
}
void Linear::save(std::ofstream& f) const{W1->save(f);B1->save(f);}
void Linear::load(std::ifstream& f){W1->load(f);B1->load(f);}
graph Linear::forward(const graph & X)
{   
        if(X->dim[3] != W1->dim[2])
        {
            std::cout << "Shape Mismatch in Linear Layer of " << X->op_name <<": \n";
            std::cout << "Dimensions are input:  (" << X->dim[2] << "," << X->dim[3] << ") and (" << W1->dim[2] << ","<<W1->dim[3] << ") \n";
            std::exit(1); 
        }

        const int batch = X->dim[0];
        const int row = X->dim[2];
        const int col = X->dim[3];
        const int tpb = THREADSPERBLOCK;
        auto node = std::make_shared<NodeBackProp>(op_name, batch, 1, row, out, 1);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((out+BLOCK_SIZE-1)/BLOCK_SIZE,(row+BLOCK_SIZE-1)/BLOCK_SIZE,batch);
        dim3 grid2((out+BLOCK_SIZE-1)/BLOCK_SIZE,(col+BLOCK_SIZE-1)/BLOCK_SIZE,batch);
        node->inputs = {X};
        
        node->forward = [=]()
        {   
            bmm<<<grid, block>>>(X->output, W1->output, node->output, batch, row, col, out,0,1,0,1); //Assignment
            CheckError("MatMul... X*W1 in Linear Layer forward");

            BCumAdd<<<(tpb+batch*row-1)/tpb, tpb>>>(node->output, B1->output,batch, row, out); //Assignment
            CheckError("Add... X*W1+B1 in Linear Layer forward");

        };

        node->backward= [=]()
        {
           
            bmmABT<<<grid, block>>>(node->grad, W1->output, X->grad, batch, row, out, W1->dim[2], 1,1,0,1);
            CheckError("MatMul... ∂Z*W^T in Linear Layer backward");

            bmmATB<<<grid2, block>>>(X->output, node->grad, W1->grad,batch,row,X->dim[3],node->dim[3],1,1,1,0);
            CheckError("MatMul... X^T*∂Z in Linear Layer backward");

            BCompress<<<(tpb+batch*node->dim[2]-1)/tpb, tpb>>>(node->grad, B1->grad, batch, node->dim[2],node->dim[3]);
            CheckError("Compress... Squeeze(∂Z)->∂b in Lineary Layer backward");
            
            
        };
        
        node->free = [=](){node->clear();};

        node->zero_grad = [=]()
        {
            Zerograd(node);
            Zerograd(W1);
            Zerograd(B1);
        };
        
        node->accumulate = [=](double* global_scale)
        {
            W1->accumulate_grad(global_scale);
            B1->accumulate_grad(global_scale);
        };

        node->clipnorm = [=](const double* global_scale)
        {
            W1->gradnorm(global_scale);
            B1->gradnorm(global_scale);
        };

        node->updateParams = [=]()
        {
            W1->update();
            B1->update();
        };

        
        node->printparams = [=]()
        {
        printHeadGPU(W1);
        printHeadGPU(B1);
        };

        return node;

}

Convolute2D::Convolute2D(GraphOperations&go_ref, int Input, int Output, int C, int D, int stride, int padding, str param) 
: go(go_ref), out(Output), inp(Input), c(C), d(D), pad(padding), stride(stride), name(param)
{   
    
        weights = new AdamParameter(name+ " Weight ", 1, out, inp, c, d);
        bias    = new AdamParameter(name+ " Bias ", 1, 1, out, 1, 1);
}
void Convolute2D::save(std::ofstream& f) const{weights->save(f); bias->save(f);}
void Convolute2D::load(std::ifstream& f){weights->load(f); bias->load(f);}
graph Convolute2D::forward(const graph& X)
{   
    if(X->dim[1] != inp)
    {
        std::cout << "Dimension Mismatch! in "<< name <<": \n"; Dimension(X);
        std::cout << "Actual input (Batch x depth): (" << X->dim[0] << "x" << X->dim[1] <<  ") \n";
        std::cout << "Expected input (Batch x depth): (" << X->dim[0] << "x" << inp <<  ") \n"; std::exit(1);
    }
    
    const int batch = X->dim[0];
    const int a = X->dim[2];  
    const int b = X->dim[3];  
    const int outR = (2 * pad + a - c) / stride + 1;
    const int outC = (2 * pad + b - d) / stride + 1;    
    const int backward_pad = c - 1 - pad;
    const int gradR_raw = (2 * backward_pad + outR - c) / stride + 1;
    const int gradC_raw = (2 * backward_pad + outC - d) / stride + 1;
    bool needs_padding = (gradR_raw != a || gradC_raw != b);
    const int padTop = (a > gradR_raw) ? (a - gradR_raw) / 2 : 0;
    const int padLeft = (b > gradC_raw) ? (b - gradC_raw) / 2 : 0;
    const int padBottom = (a > gradR_raw) ? (a - gradR_raw - padTop) : 0;
    const int padRight = (b > gradC_raw) ? (b - gradC_raw - padLeft) : 0;
    
    float* X_grad_temp = nullptr;
    size_t temp_size = batch * inp * gradR_raw * gradC_raw;
    
    if (needs_padding) {
    if (padTop < 0 || padLeft < 0 || padBottom < 0 || padRight < 0) {
        printf("ERROR: Cannot pad - computed gradient (%d x %d) is larger than input (%d x %d)\n", gradR_raw, gradC_raw, a, b);
        std::exit(1);}
        SafeCudaMalloc("X_temp_grad", X_grad_temp, temp_size);
    }
    
    auto node = std::make_shared<NodeBackProp>(name, batch, out, outR, outC, 1);
    node->inputs = {X};
    go.GB += (double)(node->total) * sizeof(float) / (pow(2,30));
    const int tpb = THREADSPERBLOCK;
    dim3 block(16, 16, 4);
    dim3 grid_forward((outC + block.x - 1) / block.x, (outR + block.y - 1) / block.y, (batch * out + block.z - 1) / block.z);
    dim3 grid_weight_grad((out + block.x - 1) / block.x, (inp + block.y - 1) / block.y, (c * d + block.z - 1) / block.z);
    dim3 grid_input_grad((gradC_raw + block.x - 1) / block.x, (gradR_raw + block.y - 1) / block.y,(batch * inp + block.z - 1) / block.z);
    
    node->forward = [=]()
    {
        isNan(X);
        CV3D<<<grid_forward, block>>>(X->output,weights->output,bias->output,node->output,batch,out,inp,a,b,c,d,pad,stride,0);
        CheckError("Forward Convolution in " + name);
    };
    
    node->backward = [=]()
    {
        isNan(node, 1);
        GV2D<<<grid_weight_grad, block>>>(X->output,node->grad,weights->grad,batch,out,inp,a,b,c,d,pad,stride);
        CheckError("Weight Gradient in " + name);
        
        Channel_Squeeze1D<<<(batch*out+tpb-1)/tpb,tpb>>>(node->grad,bias->grad,batch,out,outR,outC);
        CheckError("Bias Gradient in " + name);

        if (needs_padding) 
        {
            CV3D<<<grid_input_grad, block>>>(node->grad,weights->output,nullptr,X_grad_temp,batch,inp,out,outR,outC,c,d,backward_pad,stride,1);
            CheckError("Input Gradient (raw) in " + name);
            const long long total = (long long)batch * (long long)inp * a * b;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            PadOutput<<<blocks, threads>>>(X_grad_temp,X->grad,batch,inp,gradR_raw,gradC_raw,a,b,padTop,padLeft);
            CheckError("Padding Input Gradient in " + name);
            
        } 

        else 
        {
            CV3D<<<grid_input_grad, block>>>(node->grad,weights->output,nullptr,X->grad,batch,inp,out,outR,outC,c,d,backward_pad,stride,1);
            CheckError("Input Gradient in " + name);
        }

    };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd(weights);
        Zerograd(bias);
        if (needs_padding && X_grad_temp != nullptr) Zerograd("X_grad_temp", X_grad_temp, temp_size);
    };
    
    node->free = [=]()
    {
        if(X_grad_temp != nullptr) cudaFree(X_grad_temp);
        node->clear();
    };
    
    node->accumulate = [=](double* global_scale)
    {
        weights->accumulate_grad(global_scale);
        bias->accumulate_grad(global_scale);
    };
    
    node->clipnorm = [=](const double* global_scale)
    {
        weights->gradnorm(global_scale);
        bias->gradnorm(global_scale);
    };
    
    node->updateParams = [=]()
    {
        weights->update();
        bias->update();
    };
    
    node->printparams = [=]()
    {
        printHeadGPU(weights);
        printHeadGPU(bias);
    };

    return node;
}

/*
Con2D::Conv2D(GraphOperations& go_ref, int Input, int Output, int C, int D, int s, int p, std::string param) 
    : go(go_ref), out(Output), inp(Input), c(C), d(D), stride(s), pad(p), name(param) {
    
    weights = new AdamParameter(name + " Weight ", 1, out, inp, c, d);
    bias    = new AdamParameter(name + " Bias ", 1, 1, out, 1, 1);

    // Initialize cuDNN
    cudnnCreate(&cudnn);
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set permanent descriptors
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out, inp, c, d);
    cudnnSetConvolution2dDescriptor(conv_desc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out, 1, 1);
}

Conv2D::~Conv2D() 
{
    if (d_workspace) cudaFree(d_workspace);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroy(cudnn);
}
void Conv2D::ValidateCache(int n, int c_in, int h, int w) 
{
    if (cache.n == n && cache.c == c_in && cache.h == h && cache.w == w) return;

    cache.n = n; cache.c = c_in; cache.h = h; cache.w = w;
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c_in, h, w);

    int outN, outC, outH, outW;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc, &outN, &outC, &outH, &outW);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outN, outC, outH, outW);

    // Benchmarking
    int returnedAlgoCount;
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    cudnnFindConvolutionForwardAlgorithm(cudnn, x_desc, w_desc, conv_desc, y_desc, 1, &returnedAlgoCount, &fwd_perf);
    cache.fwd_algo = fwd_perf.algo;

    cudnnConvolutionBwdDataAlgoPerf_t bwd_data_perf;
    cudnnFindConvolutionBackwardDataAlgorithm(cudnn, w_desc, y_desc, conv_desc, x_desc, 1, &returnedAlgoCount, &bwd_data_perf);
    cache.bwd_data_algo = bwd_data_perf.algo;

    cudnnConvolutionBwdFilterAlgoPerf_t bwd_filter_perf;
    cudnnFindConvolutionBackwardFilterAlgorithm(cudnn, x_desc, y_desc, conv_desc, w_desc, 1, &returnedAlgoCount, &bwd_filter_perf);
    cache.bwd_filter_algo = bwd_filter_perf.algo;

    // Workspace Allocation
    size_t s_fwd, s_data, s_filter;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, x_desc, w_desc, conv_desc, y_desc, cache.fwd_algo, &s_fwd);
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, w_desc, y_desc, conv_desc, x_desc, cache.bwd_data_algo, &s_data);
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, x_desc, y_desc, conv_desc, w_desc, cache.bwd_filter_algo, &s_filter);
    
    cache.workspace_size = std::max({s_fwd, s_data, s_filter});
    if (d_workspace) cudaFree(d_workspace);
    cudaMalloc(&d_workspace, cache.workspace_size);
}

void Conv2D::save(std::ofstream& f) const{weights->save(f); bias->save(f);}
void Conv2D::load(std::ifstream& f)  {weights->load(f); bias->load(f);}

graph Conv2D::forward(const graph& X) 
{
    ValidateCache(X->dim[0], X->dim[1], X->dim[2], X->dim[3]);
    int n, c_out, outH, outW;
    cudnnGetTensor4dDescriptor(y_desc, nullptr, &n, &c_out, &outH, &outW, nullptr, nullptr, nullptr, nullptr);
    auto node = std::make_shared<NodeBackProp>(name, n, out, outH, outW, 1);
    node->inputs = {X};

    node->forward = [=]() 
    {
        float alpha = 1.0f, beta = 0.0f;
        cudnnConvolutionForward(cudnn, &alpha, x_desc, X->output, w_desc, weights->output, 
                                conv_desc, cache.fwd_algo, d_workspace, cache.workspace_size, 
                                &beta, y_desc, node->output);
        cudnnAddTensor(cudnn, &alpha, bias_desc, bias->output, &alpha, y_desc, node->output);
    };

    node->backward = [=]() 
    {
        float alpha = 1.0f, beta_acc = 1.0f; 
        cudnnConvolutionBackwardFilter(cudnn, &alpha, x_desc, X->output, y_desc, node->grad, 
                                       conv_desc, cache.bwd_filter_algo, d_workspace, cache.workspace_size, 
                                       &beta_acc, w_desc, weights->grad);

        cudnnConvolutionBackwardBias(cudnn, &alpha, y_desc, node->grad, &beta_acc, bias_desc, bias->grad);
        cudnnConvolutionBackwardData(cudnn, &alpha, w_desc, weights->output, y_desc, node->grad, 
                                     conv_desc, cache.bwd_data_algo, d_workspace, cache.workspace_size, 
                                     &beta_acc, x_desc, X->grad);
    };

    node->zero_grad = [=](){Zerograd(node);Zerograd(weights);Zerograd(bias);};
    
    node->free = [=](){node->clear();};
    
    node->accumulate = [=](double* global_scale)
    {weights->accumulate_grad(global_scale);bias->accumulate_grad(global_scale);};
    
    node->clipnorm = [=](const double* global_scale)
    {weights->gradnorm(global_scale);bias->gradnorm(global_scale);};
    
    node->updateParams = [=](){weights->update();bias->update();};
    return node;
}
    
*/

Convolute2DT::Convolute2DT(GraphOperations& go_ref, int Input, int Output, int C, int D, int stride, int padding, str param) 
    : go(go_ref), out(Output), inp(Input), c(C), d(D), pad(padding), stride(stride), name(param)
{   
    weights = new AdamParameter(name + " Weight ", 1, out, inp, c, d);
    bias    = new AdamParameter(name + " Bias ", 1, 1, out, 1, 1);
}
void Convolute2DT::save(std::ofstream& f) const{weights->save(f); bias->save(f);}
void Convolute2DT::load(std::ifstream& f) {weights->load(f); bias->load(f);}
graph Convolute2DT::forward(const graph& X)
{   
    if(X->dim[1] != inp)
    {
        std::cout << "Dimension Mismatch! in " << name << ": \n";
        Dimension(X);
        std::cout << "Actual input (Batch x depth): (" << X->dim[0] << "x" << X->dim[1] <<  ") \n";
        std::cout << "Expected input (Batch x depth): (" << X->dim[0] << "x" << inp <<  ") \n";
        std::exit(1);
    }
    
    const int batch = X->dim[0];
    const int inp_h = X->dim[2];
    const int inp_w = X->dim[3];
    const int out_h = (inp_h - 1) * stride - 2 * pad + c;
    const int out_w = (inp_w - 1) * stride - 2 * pad + d;
    
    auto node = std::make_shared<NodeBackProp>(name, batch, out, out_h, out_w, 1);
    node->inputs = {X};
    go.GB += (double)(node->total) * sizeof(float) / (pow(2, 30));
    
    const int tpb = THREADSPERBLOCK;
    const size_t smem_wgrad = tpb * sizeof(float);
    dim3 block(16, 16, 4);
    dim3 grid_fwd((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,(batch * out + block.z - 1) / block.z);
    dim3 grid_wgrad(out * inp, c, d);  
    dim3 grid_igrad((inp_w + block.x - 1)/block.x,(inp_h + block.y - 1) / block.y, (batch * inp + block.z - 1) / block.z);

    node->forward = [=]()
    {

        isNan(X);
        CVT2D_Forward<<<grid_fwd, block>>>(X->output, weights->output, bias->output, node->output, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        CheckError("ConvTranspose2D Forward Kernel");
        isNan(node);
    };

    node->backward = [=]()
    {
        isNan(node, 1);
        CVT2D_GradWeights<<<grid_wgrad,tpb,smem_wgrad>>>(X->output, node->grad, weights->grad, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        CheckError("Gradient w.r.t Kernels for ConvTranspose2D");

        Channel_Squeeze1D<<<(batch*out+tpb-1)/tpb,tpb>>>(node->grad, bias->grad, batch, out, out_h, out_w);
        CheckError("Squeeze in ConvTranspose2D backward");

        CVT2D_GradInput<<<grid_igrad, block>>>(node->grad, weights->output, X->grad, batch, out, inp, inp_h, inp_w, c, d, pad, stride);
        CheckError("Gradient w.r.t input for ConvTranspose2D");
    };

    node->zero_grad = [=]()
    {
        Zerograd(node);
        Zerograd(weights); 
        Zerograd(bias);
    };

    node->free = [=]()
    {
        node->clear();
        weights->clear();
        bias->clear();
    };

    node->accumulate = [=](double* global_scale)
    {
        weights->accumulate_grad(global_scale);
        bias->accumulate_grad(global_scale);
    };

    node->clipnorm = [=](const double* global_scale)
    {
        weights->gradnorm(global_scale);
        bias->gradnorm(global_scale);
    };

    node->updateParams = [=]()
    {
        weights->update();
        bias->update();
    };

    node->printparams = [=]()
    {
        printHeadGPU(weights);
        printHeadGPU(bias);
    };

    return node;
}

VisionAttention::VisionAttention(GraphOperations &go_ref, const int Channels): go(go_ref), channels(Channels)
{
        Q = new Convolute2D(subGraph, channels, channels,1,1,1,0, "Q"); 
        K = new Convolute2D(subGraph, channels, channels,1,1,1,0, "K");
        V = new Convolute2D(subGraph, channels, channels,1,1,1,0, "V");
        P = new Convolute2D(subGraph, channels, channels,1,1,1,0, "P");
    

}
void VisionAttention::save(std::ofstream& f) const{Q->save(f); K->save(f); V->save(f); P->save(f);}
void VisionAttention::load(std::ifstream& f){Q->load(f); K->load(f); V->load(f); P->load(f);}
graph VisionAttention::forward(const graph& X_in)
{
    if(X_in->dim[1] != channels)
    {
        std::cout << "Shape mismatch in Attention::forward()... Expected: " 
        << channels << "channels | \t " << " Received: " << X_in->dim[1] << " channels \n";
    }
        
    auto query = Q->forward(X_in);
    auto key = K->forward(X_in);     
    auto value = V->forward(X_in);   

    std::vector<int>init = {key->dim[0], key->dim[1], key->dim[2], key->dim[3]};
    std::vector<int>reshaped = {key->dim[0],1, key->dim[1], key->dim[2]*key->dim[3]};
    query->reshape(reshaped); key->reshape(reshaped); value->reshape(reshaped);
    auto pquery = go.Permute(query,0,1,3,2); // batch x seq_len x channels
    auto pkey = go.Permute(key,0,1,3,2);     // batch x seq_len x channels
    auto attn_scores = go.BMMABT(pquery,pkey); // batch x seq_len x seq_len
    auto scaled_attn = go.Scale(attn_scores, 1.0f/sqrtf((float)channels));
    auto softmax_attn = go.SOFTMAX(scaled_attn, 1); // batch x seq_len x seq_len
    auto attn_output = go.BMM(softmax_attn,go.Permute(value, 0,1,3,2)); // batch x seq_len x channels
    auto p_attn_output = go.Permute(attn_output,0,1,3,2); // batch x channels x seq_len
    p_attn_output->reshape(init);
    auto project = P->forward(p_attn_output); 
    auto output = go.Add(X_in, project); output->op_name = "Attention Output";
    return output;
}

VisionCrossAttention::VisionCrossAttention(GraphOperations &go_ref,const int Channels, const int ContextLen, const int EmbedDim): 
go(go_ref), channels(Channels), context_len(ContextLen), embed_dim(EmbedDim)
{   
        if(channels != embed_dim)
        {
            std::cout << "ERROR: channels (" << channels << ") must equal embed_dim ("<< embed_dim << ") for cross-attention\n";
            std::exit(1);
        }
        Q = new Convolute2D(subGraph, channels,  channels, 1, 1, 1, 0, "Q");
        K = new Convolute2D(subGraph, embed_dim,embed_dim, 1, 1, 1, 0, "K");
        V = new Convolute2D(subGraph, embed_dim,embed_dim, 1, 1, 1, 0, "V");
        P = new Convolute2D(subGraph, channels,  channels, 1, 1, 1, 0, "P");
}
void VisionCrossAttention::save(std::ofstream& f) const{Q->save(f); K->save(f); V->save(f); P->save(f);}
void VisionCrossAttention::load(std::ifstream& f) {Q->load(f); K->load(f); V->load(f); P->load(f);}
graph VisionCrossAttention::forward(const graph& X_in, const graph& Context)
{   
        if(Context == nullptr)
        {
            std::cout << "Context node is null in VisionCrossAttention::forward()\n";
            std::exit(1);
        }

        if(X_in->dim[1] != channels)
        {
            std::cout << "Shape mismatch in VisionCrossAttention::forward()... Expected: " << batch << " x " << channels 
                      << " Received: " << X_in->dim[0] << " x " << X_in->dim[1] << "\n";
            std::exit(1);
        }

        if(Context->dim[0] != X_in->dim[0])
        {
            std::cout << "Context batch mismatch in AttentionT::forward()\n";
            std::exit(1);
        }
        
        if(embed_dim != channels)
        {
            std::cout << "Embed dim and channels must be equal in VisionCrossAttention::forward()\n";
            std::exit(1);
        }

        auto query = Q->forward(Context); // batch x embed dim == channels x seq_len
        auto key = K->forward(X_in);  // batch x channels x seq_len
        auto value = V->forward(X_in); // batch x channels x seq_len
        
        std::vector<int> init = {key->dim[0], key->dim[1], key->dim[2], key->dim[3]};
        std::vector<int> reshaped = {key->dim[0],1, key->dim[1], key->dim[2]*key->dim[3]};
        query->reshape(reshaped); key->reshape(reshaped); value->reshape(reshaped);

        auto pquery = go.Permute(query,0,1,3,2); // batch x seq_len x channels
        auto pkey = go.Permute(key,0,1,3,2);     // batch x seq_len x channels
        auto attn_scores = go.BMMABT(pquery,pkey); // batch x seq_len x seq_len
        auto scaled_attn = go.Scale(attn_scores, 1.0f/sqrtf((float)channels));
        auto softmax_attn = go.SOFTMAX(scaled_attn, 1); // batch x seq_len x seq_len
        auto attn_output = go.BMM(softmax_attn,go.Permute(value,0,1,3,2)); // batch x seq_len x channels
        auto p_attn_output = go.Permute(attn_output,0,1,3,2); // batch x channels x seq_len
        p_attn_output->reshape(init);
        auto project = P->forward(p_attn_output);
        auto output = go.Add(X_in, project); output->op_name = "Cross-Attention Output";
        return output;
        


}

void diffuse(float* input, float* model, float* theta, const long long total, const int t, const int T, const double s, const uint64_t seed)
{
    const double t_scale = ((double)t / T) + s;
    const double t_scale_1 = ((double)(t-1.0) / T) + s;
    const double cost_t = cos((t_scale/(1.0+s))*PIBY2);
    const double cost_t_1 = cos((t_scale_1/(1.0+s))*PIBY2);
    const double cost_t_b = cos((s/(1.0 + s))*PIBY2);
    const double alpha_hat = (cost_t*cost_t)/(cost_t_b*cost_t_b); 
    const double alpha_hat_1 = (cost_t_1*cost_t_1)/(cost_t_b*cost_t_b); 
    const double alpha_t = alpha_hat / alpha_hat_1; //  a_t
    const double beta_t = 1.0 - alpha_t;  // b_t
    const double scale_out = 1.0 / sqrt(alpha_t);
    const double scale_mean = beta_t / sqrt(1.0-alpha_hat);
    const int tpb = THREADSPERBLOCK;
    const int bpg = (tpb + total-1) / tpb;
    const double sqrt_beta = sqrt(beta_t);
    ScaleAdd<<<bpg, tpb>>>(input, model, theta, -scale_mean, total);
    CheckError("Scale Subtract in Diffuse u_0 = X_t - (B_t / sqrt(1-a_hat))*e_0(X_t,t)");

    ScaleValue(theta, scale_out, total);
    CheckError(" Scale in Diffuse 1/sqrt(a_t)*u_0");

    if(t > 1) ReplaceNoise<<<bpg, tpb>>>(input, theta, sqrt_beta, total, seed);
    else cudaMemcpy(input, theta, total*sizeof(float), cudaMemcpyDeviceToDevice);
    CheckError(" X_t = U_0(t) + sqrt(b_t)*N(0,1)");
}

void Noise(const graph & input)
{
    std::random_device rd;
    const uint64_t seed =  ((uint64_t)rd() << 32) | rd();
    GaussianNoise<<<(input->total+THREADSPERBLOCK-1)/THREADSPERBLOCK,THREADSPERBLOCK>>>(input->output, input->total, seed);
    CheckError("Addition of Gaussian noise in noise kernel");
}
