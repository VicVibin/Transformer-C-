#include "includes/debugging_utils.h"

Timing::Timing(const str reason): function(reason){};

Timing::~Timing(){}

void Timing::start(){beg = std::chrono::high_resolution_clock::now();}

void Timing::end()
{   ending = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(ending - beg);
    std::cout << "Time elapsed for " << function << ": " << duration.count() << "ms \n";
}

Text ImagePaths(const std::string& folder, const int filenums ) 
{
    Text files;

    str search_path = folder + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(search_path.c_str(), &fd);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.emplace_back(fd.cFileName);
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }

    if (files.size() < filenums)
    {
        std::cout << "Not enough files in directory... num files: " << files.size() << "\n";
        if(files.size() > 0 ) for(const auto &i : files) std::cout << i << "\t";
    }

    if(files.size() == filenums)
    {
        return files;
    }

    Text new_files;
    for(int i =0; i < filenums;++i) new_files.push_back(files[i]);

    return new_files;
}

void CheckError(const str& reason)
{
    cudaError_t launchErr = cudaGetLastError();     // did launch succeed?
    cudaError_t syncErr   = cudaDeviceSynchronize(); // did kernel succeed?

    if (launchErr != cudaSuccess) {
    std::cerr << "Launch error in " << reason << ": " << cudaGetErrorString(launchErr) << std::endl;std::exit(1); }

    if (syncErr != cudaSuccess) {
    std::cerr << "Runtime error in " << reason << ": " << cudaGetErrorString(syncErr) << std::endl; std::exit(1);
    }
}


