#include "includes/word2vec.h"


text Load(const str& path)
{
    TextProcessor processor;
    text data = processor.readAllStories(path);
    text cleanedWords = processor.cleanText(data);
    return cleanedWords;
}

Tokenization tokenizer(const text & Stories, int token_length) 
{
    Tokenization result;
    std::unordered_set<str> seen;
    Math operations;
    int t = 1;
    int size = 1;
    for (const auto& word : Stories) {
        if (seen.insert(word).second) 
        {  // insert returns true if word was not present
            result.word_map[word] = operations.random_vector(token_length, size);
            result.index[word] = t;
            t++;
        }
        
    }

    result.word_map["<start>"] = operations.random_vector(token_length, size);
    result.word_map["<end>"] = operations.random_vector(token_length, size);
    result.index["<start>"] = 0;
    result.index["<end>"] = result.word_map.size() - 1;
    return result;
}

Matrix_d text2matrix(dictionary data, text story)
{
    Matrix_d matrix;
    for(const str& word : story)
    {
        matrix.push_back(data[word]);
    }
    return matrix;
}

Matrix_d PEncoding(Matrix_d embedding)
{
    Math math;
    Linear_Algebra lin;
    int d_model = embedding[0].size();
    int seq_len = embedding.size();
    float ln_10 = math.ln(10);
    Matrix_d matrix;

    for (int pos = 0; pos < seq_len; pos++)
    {
        vector_d pos_encoding(d_model, 0.0f);

        for (int i = 0; i < d_model; i++)
        {
            float angle = pos / math.exp((8.0 * ln_10) / d_model);

            if (i % 2 == 0)
                pos_encoding[i] = math.sin(angle);
            else
                pos_encoding[i] = math.cos(angle);
        }

        embedding[pos] = lin.vectorsum(embedding[pos], pos_encoding);
    }

    return embedding;
}

Matrix_d decoder_input(dictionary data, text story)
{
    Matrix_d matrix;
    matrix.push_back(data["<start>"]);
    for(const auto &word : story)
    {
        matrix.push_back(data[word]);
    }
    return matrix;
}

Matrix_d encoder_input(dictionary data, text story)
{
    Matrix_d matrix;
    for(const str& word : story)
    {
        matrix.push_back(data[word]);
    }
    return matrix;
}

void vec_replace(dictionary& data, text& story, const Matrix_d & update) 
{   
    for(int i = 0; i < story.size(); i++)
    {
        data[story[i]] = update[i];
    }

}

Matrix_d one_hot_encoding(indicing data, text story)
{   
    int dic_size = data.size();
    int story_size = story.size();
    Matrix_d one_hot(story_size, vector_d(dic_size, 0.0f));
    for(int i = 0; i < story_size; i++)
    {
        int j = data[story[i]];
        one_hot[i][j] = 1.0f;
    }
    return one_hot;
}

text read_words(text stories, int a , int b)
{
    text sample;
    for(int i = a; i < b; i++)
    {
        sample.push_back(stories[i]);
    }
    return sample;
}