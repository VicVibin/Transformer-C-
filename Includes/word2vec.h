#pragma once
#include "DataLoader.h"
#include "MathGPU.h"
#include <unordered_map>


using dictionary = std::map<str, vector_d>;
using indicing = std::unordered_map<str,int>;

struct Tokenization
{
    dictionary word_map;
    std::unordered_map<str, int> index;
};

text Load(const str& path );
Matrix_d PEncoding(Matrix_d embedding);
Matrix_d text2matrix(dictionary data, text story);
Tokenization tokenizer(const text & Stories, int token_length);
Matrix_d decoder_input(dictionary data, text story);
Matrix_d encoder_input(dictionary data, text story);
void vec_replace(dictionary data, text story, const Matrix_d update);
Matrix_d one_hot_encoding(indicing data, text story);
text read_words(text Stories, int a, int b);
