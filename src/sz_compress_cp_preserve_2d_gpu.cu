#include "sz_cp_preserve_utils.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include "sz3_utils.hpp"
#include "sz_lossless.hpp"
#include "utils.hpp"
//#include "sz_compress_3d.hpp"
//#include "sz_compress_cp_preserve_2d.hpp"
#include <chrono>
#include <thrust/execution_policy.h>
#include <thrust/scatter.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

using namespace std;

#include "kernel/lrz/lproto.hh"

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 8
#define NUM_PRE_THREAD 4

// replace dEb_U[id] == 0 to replace_num
template<typename T>
struct ReplaceZero {
    T replace_num;
    
    __host__ __device__
    ReplaceZero(T replace_num) : replace_num(replace_num) {}

    __host__ __device__
    T operator()(T x) const {
        return (x == 0) ? replace_num : x;
    }
};

template<typename T>
struct IsZero {
    __host__ __device__
    IsZero() {}

    __host__ __device__
    bool operator()(T x) const {
        return x == 0;
    }
};

template<typename T>
[[nodiscard]] constexpr inline T max_eb_to_keep_sign_2d_offline_2(const T volatile u0, const T volatile u1, const int degree=2){
    T positive = 0;
    T negative = 0;
    accumulate(u0, positive, negative);
    accumulate(u1, positive, negative);
    return max_eb_to_keep_sign(positive, negative, degree);
}

template<typename T>
[[nodiscard]] constexpr inline T gpu_max_eb_to_keep_sign_2d_offline_2_degree2(const T u0, const T u1){
    //if(value >= 0) positive += value;
	// else negative += - value;
    T positive = (u0>=0 ? u0 : 0) + (u1>=0? u1 : 0);
    T negative = (u0<0 ? -u0 : 0) + (u1<0? -u1 : 0);
    T P = sqrt(positive);
    T N = sqrt(negative);
    return fabs(P - N)/(P + N);
}

template<typename T>
[[nodiscard]] constexpr inline T max_eb_to_keep_sign_2d_offline_4(const T volatile u0, const T volatile u1, const T volatile u2, const T volatile u3, const int degree=2){
    T positive = 0;
    T negative = 0;
    accumulate(u0, positive, negative);
    accumulate(u1, positive, negative);
    accumulate(u2, positive, negative);
    accumulate(u3, positive, negative);
    return max_eb_to_keep_sign(positive, negative, degree);
}

template<typename T>
[[nodiscard]] constexpr inline T gpu_max_eb_to_keep_sign_2d_offline_4_degree2(const T u0, const T u1, const T u2, const T u3){
    T positive = (u0>=0 ? u0 : 0) + (u1>=0? u1 : 0) + (u2>=0 ? u2 : 0) + (u3>=0? u3 : 0);
    T negative = (u0<0 ? -u0 : 0) + (u1<0? -u1 : 0) + (u2<0 ? -u2 : 0) + (u3<0? -u3 : 0);
    T P = sqrt(positive);
    T N = sqrt(negative);
    return fabs(P - N)/(P + N);
}


template<typename T>
[[nodiscard]] constexpr inline double max_eb_to_keep_position_and_type(const T volatile u0, const T volatile u1, const T volatile u2, const T volatile v0, const T volatile v1, const T volatile v2)
{	
    T u0v1 = u0 * v1;
    T u1v0 = u1 * v0;
    T u0v2 = u0 * v2;
    T u2v0 = u2 * v0;
    T u1v2 = u1 * v2;
    T u2v1 = u2 * v1;
    T det = u0v1 - u1v0 + u1v2 - u2v1 + u2v0 - u0v2;
    T eb = 0;
    if(det != 0){
        bool f1 = (det / (u2v0 - u0v2) >= 1);//u0v1 - u1v0 + u1v2 - u2v1>=0
        bool f2 = (det / (u1v2 - u2v1) >= 1);//u0v1 - u1v0 + u2v0 - u0v2>=0
        bool f3 = (det / (u0v1 - u1v0) >= 1);//u1v2 - u2v1 + u2v0 - u0v2>=0
        if(f1 && f2 && f3){
            eb=0;
        }
        else{
            // no critical point
            eb = 0;
            if(!f1){
                T eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u2v0, -u0v2), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u1v2, -u2v1));
                // double eb_cur = MINF(max_eb_to_keep_sign_2(u2, u0, v2, v0), max_eb_to_keep_sign_4(u0, u1, u2, v0, v1, v2));
                eb = MAX(eb, eb_cur);
            }
            if(!f2){
                T eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u1v2, -u2v1), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u2v0, -u0v2));
                // double eb_cur = MINF(max_eb_to_keep_sign_2(u1, u2, v1, v2), max_eb_to_keep_sign_4(u2, u0, u1, v2, v0, v1));
                eb = MAX(eb, eb_cur);
            }
            if(!f3){
                T eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u0v1, -u1v0), max_eb_to_keep_sign_2d_offline_4(u1v2, -u2v1, u2v0, -u0v2));
                // double eb_cur = MINF(max_eb_to_keep_sign_2(u0, u1, v0, v1), max_eb_to_keep_sign_4(u1, u2, u0, v1, v2, v0));
                eb = MAX(eb, eb_cur);
            }
            // eb = MINF(eb, DEFAULT_EB);
        }
    }
    return eb;
}

template<typename T>
[[nodiscard]] constexpr inline double gpu_max_eb_to_keep_position_and_type(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2)
{
    auto gpu_minf = [](auto a, auto b) -> T{ return (a<b)?a:b; };
#define U0V1 u0*v1
#define U1V0 u1*v0
#define U0V2 u0*v2
#define U2V0 u2*v0
#define U1V2 u1*v2
#define U2V1 u2*v1
    //T u0v1 = u0 * v1;
    //T u1v0 = u1 * v0;
    //T u0v2 = u0 * v2;
    //T u2v0 = u2 * v0;
    //T u1v2 = u1 * v2;
    //T u2v1 = u2 * v1;
    //T det = u0v1 - u1v0 + u1v2 - u2v1 + u2v0 - u0v2;
    T det = U0V1 - U1V0 + U1V2 - U2V1 + U2V0 - U0V2;
    T eb = 0;
    if(det != 0)
    {   
        //T d1 = u2v0 - u0v2;
        //T d2 = u1v2 - u2v1;
        //T d3 = u0v1 - u1v0;
        T d1 = U2V0 - U0V2;
        T d2 = U1V2 - U2V1;
        T d3 = U0V1 - U1V0;
        bool f1 = (det / d1 >= T(1));
        bool f2 = (det / d2 >= T(1));
        bool f3 = (det / d3 >= T(1)); 
        if(!f1){
            // T eb_cur = gpu_minf(gpu_max_eb_to_keep_sign_2d_offline_2_degree2(U2V0, -U0V2), gpu_max_eb_to_keep_sign_2d_offline_4_degree2(U0V1, -U1V0, U1V2, -U2V1));
            
            T pos1 = (U2V0 >= 0 ? U2V0 : 0) + ((-U0V2) >= 0 ? (-U0V2) : 0);
            T neg1 = (U2V0 < 0 ? -U2V0 : 0) + ((-U0V2) < 0 ? U0V2 : 0);
            T P1 = sqrt(pos1);
            T N1 = sqrt(neg1);
            T res1 = fabs(P1 - N1) / (P1 + N1);

            T pos2 = (U0V1 >= 0 ? U0V1 : 0) + ((-U1V0) >= 0 ? (-U1V0) : 0) + (U1V2 >= 0 ? U1V2 : 0) + ((-U2V1) >= 0 ? (-U2V1) : 0);
            T neg2 = (U0V1 < 0 ? -U0V1 : 0) + ((-U1V0) < 0 ? U1V0 : 0) + (U1V2 < 0 ? -U1V2 : 0) + ((-U2V1) < 0 ? U2V1 : 0);
            T P2 = sqrt(pos2);
            T N2 = sqrt(neg2);
            T res2 = fabs(P2 - N2) / (P2 + N2);
            T eb_cur = gpu_minf(res1, res2);

            eb = MAX(eb, eb_cur);
        }
        if(!f2){
            // T eb_cur = gpu_minf(gpu_max_eb_to_keep_sign_2d_offline_2_degree2(U1V2, -U2V1), gpu_max_eb_to_keep_sign_2d_offline_4_degree2(U0V1, -U1V0, U2V0, -U0V2));
            
            T pos1 = (U1V2 >= 0 ? U1V2 : 0) + ((-U2V1) >= 0 ? (-U2V1) : 0);
            T neg1 = (U1V2 < 0 ? -U1V2 : 0) + ((-U2V1) < 0 ? U2V1 : 0);
            T P1 = sqrt(pos1);
            T N1 = sqrt(neg1);
            T res1 = fabs(P1 - N1) / (P1 + N1);

            T pos2 = (U0V1 >= 0 ? U0V1 : 0) + ((-U1V0) >= 0 ? (-U1V0) : 0) + (U2V0 >= 0 ? U2V0 : 0) + ((-U0V2) >= 0 ? (-U0V2) : 0);
            T neg2 = (U0V1 < 0 ? -U0V1 : 0) + ((-U1V0) < 0 ? U1V0 : 0) + (U2V0 < 0 ? -U2V0 : 0) + ((-U0V2) < 0 ? U0V2 : 0);
            T P2 = sqrt(pos2);
            T N2 = sqrt(neg2);
            T res2 = fabs(P2 - N2) / (P2 + N2);
            T eb_cur = gpu_minf(res1, res2);
            
            eb = MAX(eb, eb_cur);
        }
        if(!f3){
            // T eb_cur = gpu_minf(gpu_max_eb_to_keep_sign_2d_offline_2_degree2(U0V1, -U1V0), gpu_max_eb_to_keep_sign_2d_offline_4_degree2(U1V2, -U2V1, U2V0, -U0V2));
            T pos1 = (U0V1 >= 0 ? U0V1 : 0) + ((-U1V0) >= 0 ? (-U1V0) : 0);
            T neg1 = (U0V1 < 0 ? -U0V1 : 0) + ((-U1V0) < 0 ? U1V0 : 0);
            T P1 = sqrt(pos1);
            T N1 = sqrt(neg1);
            T res1 = fabs(P1 - N1) / (P1 + N1);

            T pos2 = (U1V2 >= 0 ? U1V2 : 0) + ((-U2V1) >= 0 ? (-U2V1) : 0) + (U2V0 >= 0 ? U2V0 : 0) + ((-U0V2) >= 0 ? (-U0V2) : 0);
            T neg2 = (U1V2 < 0 ? -U1V2 : 0) + ((-U2V1) < 0 ? U2V1 : 0) + (U2V0 < 0 ? -U2V0 : 0) + ((-U0V2) < 0 ? U0V2 : 0);
            T P2 = sqrt(pos2);
            T N2 = sqrt(neg2);
            T res2 = fabs(P2 - N2) / (P2 + N2);
            T eb_cur = gpu_minf(res1, res2);

            eb = MAX(eb, eb_cur);
        }
        eb = gpu_minf(eb, 1);
    }
    return eb;
}

//version 3, single thread muti-compute 32*32 data mapto 32*8
template <typename T, int TileDim_X = BLOCKSIZE_X, int TileDim_Y = BLOCKSIZE_Y>
__global__ void derive_eb_offline_v3(const T* __restrict__ dU, const T* __restrict__ dV, T* __restrict__ dEb, T* __restrict__  dEb_U,  T* __restrict__ dEb_V, int r1, int r2, T max_pwr_eb){
    constexpr auto YSEQ = TileDim_X / TileDim_Y;
    __shared__ T buf_U[TileDim_Y * YSEQ][TileDim_X+1];
    __shared__ T buf_V[TileDim_Y * YSEQ][TileDim_X+1];
    __shared__ T per_cell_eb_L[TileDim_Y * YSEQ][TileDim_X+1];
    __shared__ T per_cell_eb_U[TileDim_Y * YSEQ][TileDim_X+1];  
    __shared__ T buf_eb[TileDim_Y * YSEQ][TileDim_X+1]; 
    //int row = blockIdx.y * (YSEQ * blockDim.y - 2) + threadIdx.y * YSEQ; // global row index
    //int col = blockIdx.x * (blockDim.x-2) + threadIdx.x; // global col index
#define row (blockIdx.y * (YSEQ * blockDim.y - 2) + threadIdx.y * YSEQ + i)
#define col blockIdx.x * (blockDim.x-2) + threadIdx.x
    //int localRow = threadIdx.y; // local row index
    //int localCol = threadIdx.x; // local col index
#define localRow (threadIdx.y*YSEQ + i)
#define localCol threadIdx.x

    for (int i = 0; i < YSEQ; i++)
    {
        buf_eb[localRow][localCol] = max_pwr_eb;
    }
    __syncthreads();

    // load data from global memory to shared memory
    for (int i = 0; i < YSEQ; i++)
        {
        if(row < r1 && col < r2){
            buf_U[localRow][localCol] = dU[row * r2 + col];
            buf_V[localRow][localCol] = dV[row * r2 + col];
        }
    }
    __syncthreads();

    for (int i = 0; i < YSEQ; i++)
    {
        if(localRow<YSEQ*TileDim_Y-1 && localCol<TileDim_X-1){
            per_cell_eb_U[localRow][localCol] = gpu_max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow][localCol+1], buf_U[localRow+1][localCol+1],
                buf_V[localRow][localCol], buf_V[localRow][localCol+1], buf_V[localRow+1][localCol+1]);
            per_cell_eb_L[localRow][localCol] = gpu_max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow+1][localCol], buf_U[localRow+1][localCol+1],
                buf_V[localRow][localCol], buf_V[localRow+1][localCol], buf_V[localRow+1][localCol+1]);
        }
        
    }
    __syncthreads();

    T localmin;
    for (int i = 0; i < YSEQ; i++)
    {
        if(localRow<YSEQ*TileDim_Y-2 && localCol<TileDim_X-2)
        {
            localmin = buf_eb[localRow][localCol];
            auto temp = per_cell_eb_U[localRow][localCol];
            localmin = min(localmin, temp);
            temp =  per_cell_eb_L[localRow][localCol];
            localmin = min(localmin, temp);
            temp =  per_cell_eb_U[localRow+1][localCol];
            localmin = min(localmin, temp);
            temp = per_cell_eb_L[localRow][localCol+1];
            localmin = min(localmin, temp);
            temp = per_cell_eb_U[localRow+1][localCol+1];
            localmin = min(localmin, temp);
            temp = per_cell_eb_L[localRow+1][localCol+1];
            localmin = min(localmin, temp);
            buf_eb[localRow][localCol] = localmin;
        }
    }
    __syncthreads();

    for (int i = 0; i < YSEQ; i++)
    {
        if(row<r1-2 && col<r2-2 && localRow<YSEQ*TileDim_Y-2 && localCol<TileDim_X-2)
        {
            T threshold = (T)(1.0 / (1 << 20));
            auto temp = buf_eb[localRow][localCol] * fabs(buf_U[localRow+1][localCol+1]);
            if(temp <= threshold) temp = 0;
            if(temp > threshold){
                uint id = log2(temp / threshold)/2.0;
                temp = (T)(1ULL << (2 * id)) * threshold;
            }
            dEb_U[(row+1) * r2 + (col+1)] = temp;

            temp = buf_eb[localRow][localCol] * fabs(buf_V[localRow+1][localCol+1]);
            if(temp <= threshold) temp = 0;
            if(temp > threshold){
                uint id = log2(temp / threshold)/2.0;
                temp = (T)(1ULL << (2 * id)) * threshold;
            }
            dEb_V[(row+1) * r2 + (col+1)] =  temp;
        }
    }
    __syncthreads();

    for (int i = 0; i < YSEQ; i++)
    {
        if((row == 0 || col ==0 || row==r1-1 || col == r2-1)&&(row<r1-1 && col<r2-1)){
            dEb_U[row * r2 + col] = 0;
            dEb_V[row * r2 + col] = 0;
        }
    }
}

// compression with pre-computed error bounds
template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_offline_gpu(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, T max_pwr_eb, 
        T* ot_val_U, uint32_t* ot_idx_U, uint32_t* ot_num_U, T* ot_val_V, uint32_t* ot_idx_V, uint32_t* ot_num_V, T* U_decomp, T* V_decomp){
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_elements = r1 * r2;
    T * eb_gpu; cudaMalloc(&eb_gpu, r1 * r2 * sizeof(T));
    cudaMemset(eb_gpu, max_pwr_eb, r2 * r1 * sizeof(T));
    T * eb = (T *) malloc(num_elements * sizeof(T));
    for(int i=0; i<num_elements; i++) eb[i] = max_pwr_eb;
    
    // CPU code
    const T * U_pos = U;
    const T * V_pos = V;
    T * eb_pos = eb;
    // coordinates for triangle_coordinates
    const T X_upper[3][2] = {{0, 0}, {1, 0}, {1, 1}};
    const T X_lower[3][2] = {{0, 0}, {0, 1}, {1, 1}};
    const size_t offset_upper[3] = {0, r2, r2+1};
    const size_t offset_lower[3] = {0, 1, r2+1};
    printf("compute eb\n");
    for(int i=0; i<r1-1; i++){
        const T * U_row_pos = U_pos;
        const T * V_row_pos = V_pos;
        float * eb_row_pos = eb_pos;
        for(int j=0; j<r2-1; j++){
            for(int k=0; k<2; k++){
                //printf("U_row_pos: %p, V_row_pos: %p, eb_row_pos: %p\n", U_row_pos, V_row_pos, eb_row_pos);
                auto X = (k == 0) ? X_upper : X_lower;
                auto offset = (k == 0) ? offset_upper : offset_lower;
                // reversed order!
                T max_cur_eb = max_eb_to_keep_position_and_type(U_row_pos[offset[0]], U_row_pos[offset[1]], U_row_pos[offset[2]],
                	V_row_pos[offset[0]], V_row_pos[offset[1]], V_row_pos[offset[2]]);
                eb_row_pos[offset[0]] = MINF(eb_row_pos[offset[0]], max_cur_eb);
                eb_row_pos[offset[1]] = MINF(eb_row_pos[offset[1]], max_cur_eb);
                eb_row_pos[offset[2]] = MINF(eb_row_pos[offset[2]], max_cur_eb);
            }
            U_row_pos ++;
            V_row_pos ++;
            eb_row_pos ++;
        }
        U_pos += r2;
        V_pos += r2;
        eb_pos += r2;
    }
    printf("compute eb done\n");
    
   
    // compression gpu
    printf("compute eb_gpu\n");   

    T *dU, *dV, *dEb_U, *dEb_V;
    cudaMalloc(&dU, r1 * r2 * sizeof(T));
    cudaMalloc(&dV, r1 * r2 * sizeof(T));
    cudaMalloc(&dEb_U, r1 * r2 * sizeof(T));
    cudaMalloc(&dEb_V, r1 * r2 * sizeof(T));
    cudaMemcpy(dU, U, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    


    cudaStream_t stream;
    cudaEvent_t a, b;

    auto bytes = 3600 * 2400 * 4 * 2.0;
    auto GiB = 1024 * 1024 * 1024.0;
    int N = 1;

    //run Kernel v3:
    dim3 blockSize_v3(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
    dim3 gridSize_v3((r2 + (blockSize_v3.x-2) - 1) / (blockSize_v3.x-2), (r1 + (blockSize_v3.y*NUM_PRE_THREAD-2)-1) / (blockSize_v3.y*NUM_PRE_THREAD-2));
    printf("gridSize_v3: %d, %d\n", gridSize_v3.x, gridSize_v3.y);
    derive_eb_offline_v3<<<gridSize_v3, blockSize_v3>>>(dU, dV, eb_gpu, dEb_U, dEb_V, r1, r2, max_pwr_eb);
    cudaDeviceSynchronize();
    printf("compute V3 eb_gpu done\n"); //
    //printf("speed GiB/s: %f\n", bytes / GiB / (ms / 1000));


    cudaStreamCreate(&stream);
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i_count=0;i_count<3;i_count++){
        float ms = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            float temp;
            cudaEventRecord(a, stream);
            derive_eb_offline_v3<<<gridSize_v3, blockSize_v3, 0, stream>>>(dU, dV, eb_gpu, dEb_U, dEb_V, r1, r2, max_pwr_eb);
            //cudaDeviceSynchronize();
            cudaEventRecord(b, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&temp, a, b);
            ms+=temp;
        }
        printf("elasped time is %f ms, V3 speed GiB/s: %f\n", ms/N, bytes / GiB / (ms / N / 1000));
    }
    cudaStreamDestroy(stream);

    //verify derive_eb
    //cpu eb_u, eb_v
    const int base = 4;
	T log2_of_base = log2(base);
	const T threshold = (T)(1.0 / (1 << 20));
    T * eb_u = (T *) malloc(num_elements * sizeof(T));
	T * eb_v = (T *) malloc(num_elements * sizeof(T));
    for(int i=0; i<num_elements; i++){
		eb_u[i] = fabs(U[i]) * eb[i];
		int temp_eb_u = eb_exponential_quantize(eb_u[i], base, log2_of_base, threshold);
		if(eb_u[i] < threshold) eb_u[i] = 0;
	}
	for(int i=0; i<num_elements; i++){
		eb_v[i] = fabs(V[i]) * eb[i];
		int temp_eb_v = eb_exponential_quantize(eb_v[i], base, log2_of_base, threshold);
		if(eb_v[i] < threshold) eb_v[i] = 0;
	}
    //verfiy eb eb_u, eb_v
    T * eb_u_gpu = (T *) malloc(num_elements * sizeof(T));;
    cudaMemcpy(eb_u_gpu, dEb_U, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    T * eb_v_gpu = (T *) malloc(num_elements * sizeof(T));;
    cudaMemcpy(eb_v_gpu, dEb_V, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    double diff = 0.0;
    double maxdiff = 0.0;
    int count=0;
    int maxdiff_index = 0;
    for (int i = 1; i < r1-1; i++){
        for(int j = 1; j < r2-1; j++){
            diff = fabs(eb_u_gpu[i*r2+j] - eb_u[i*r2+j]);
            if(diff > maxdiff)
            { 
                maxdiff = diff;
                maxdiff_index = i*r2+j;    
            }
            if (diff > std::numeric_limits<T>::epsilon()) {
                //printf("error. eb_u_gpu: %5.2f, eb_u: %5.2f,%d\n", eb_u_gpu[i],eb_u[i],i);
                //break;
                count++;
            }
        }
    }
    int count_zero = 0;
    for (int i = 0; i < r1 * r2; ++i) {
        if(eb_u_gpu[i] == 0) count_zero++;
    }
    printf("count_zero: %d\n", count_zero);
    printf("maxdiff: %f, maxdiff_index: %d, error count: %d\n", maxdiff, maxdiff_index, count);
    printf("eb_u_gpu: %f, eb_u: %f\n", eb_u_gpu[maxdiff_index], eb_u[maxdiff_index]);

    diff = 0.0;
    maxdiff = 0.0;
    count=0;
    maxdiff_index = 0;
    for (int i = 1; i < r1-1; i++){
        for(int j = 1; j < r2-1; j++){
            diff = fabs(eb_v_gpu[i*r2+j] - eb_v[i*r2+j]);
            if(diff > maxdiff)
            { 
                maxdiff = diff;
                maxdiff_index = i*r2+j;    
            }
            if (diff > std::numeric_limits<T>::epsilon()) {
                //printf("error. eb_u_gpu: %5.2f, eb_u: %5.2f,%d\n", eb_u_gpu[i],eb_u[i],i);
                //break;
                count++;
            }
        }
    }
    printf("maxdiff: %f, maxdiff_index: %d, error count: %d\n", maxdiff, maxdiff_index, count);
    printf("eb_v_gpu: %f, eb_v: %f\n", eb_v_gpu[maxdiff_index], eb_v[maxdiff_index]);

    //If U and V are both 0, means it is land in occean data, 
    //so it compress and decompress is all 0, we can just set eb = max-pwr
    thrust::counting_iterator<size_t> idx_first(0);
    thrust::counting_iterator<size_t> idx_last = idx_first + num_elements;
    thrust::for_each(
        idx_first, idx_last,
        [=] __device__ (size_t i) {
            if (dU[i] == 0 && dV[i] == 0) {
                dEb_U[i] = max_pwr_eb;
                dEb_V[i] = max_pwr_eb;
            }
        }
    );

    //deal with eb=zero  
    //U
    uint32_t* d_indices; cudaMalloc(&d_indices, r1 * r2 * sizeof(uint32_t));
    T* zero_U_data;
    uint32_t* zero_U_indices;
    cudaMalloc(&zero_U_data, r1 * r2 * sizeof(T) / 2);
    cudaMalloc(&zero_U_indices, r1 * r2 * sizeof(uint32_t) / 2);

    thrust::sequence(thrust::device, d_indices, d_indices + r1*r2);
    auto end_it = thrust::copy_if(thrust::device, dU, dU + r1*r2, dEb_U, zero_U_data, IsZero<T>());
    auto end_idx = thrust::copy_if(thrust::device, d_indices, d_indices + r1*r2, dEb_U, zero_U_indices, IsZero<T>());
    int zero_eb_U_count = end_it - zero_U_data;
    thrust::transform(thrust::device, dEb_U, dEb_U + r1*r2, dEb_U, ReplaceZero(max_pwr_eb));
    
    //V
    T* zero_V_data;
    uint32_t* zero_V_indices;
    cudaMalloc(&zero_V_data, r1 * r2 * sizeof(T) / 2);
    cudaMalloc(&zero_V_indices, r1 * r2 * sizeof(uint32_t) / 2);

    end_it = thrust::copy_if(thrust::device, dV, dV + r1*r2, dEb_V, zero_V_data, IsZero<T>());
    end_idx = thrust::copy_if(thrust::device, d_indices, d_indices + r1*r2, dEb_V, zero_V_indices, IsZero<T>());
    int zero_eb_V_count = end_it - zero_V_data;
    thrust::transform(thrust::device, dEb_V, dEb_V + r1*r2, dEb_V, ReplaceZero(max_pwr_eb));


    //comprerssion U
    float lrz_time = 0.0;  
    uint16_t *eq_U;
    cudaMalloc(&eq_U, r2 * r1 * sizeof(uint16_t));
    cudaMemset(eq_U, 0, r2 * r1 * sizeof(uint16_t));
    //compression kernel
    psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, uint16_t>(
    	dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U, dEb_U, 512, &lrz_time, 0);
    cudaDeviceSynchronize();
    //printf("ot_num_U : %d\n", *ot_num_U);   
    //comprerssion V
    uint16_t *eq_V;
    cudaMalloc(&eq_V, r2 * r1 * sizeof(uint16_t));
    cudaMemset(eq_V, 0, r2 * r1 * sizeof(uint16_t));
    //compression kernel
    psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, uint16_t>(
    	dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V, dEb_V, 512, &lrz_time, 0);
    cudaDeviceSynchronize();
    //printf("ot_num_V : %d\n", *ot_num_V);

    auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "压缩运行时间: " << elapsed.count() << " 秒" << std::endl;

    //decompression U
    T* dU_decomp; cudaMalloc(&dU_decomp, r1 * r2 * sizeof(T));
    //move ov_val to dU_decomp accrording to ot_idx
    thrust::scatter(thrust::device, ot_val_U, ot_val_U + *ot_num_U, ot_idx_U, dU_decomp);
    //decompression kernel
    psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, uint16_t>(
        eq_U, /*input*/dU_decomp, /*output*/dU_decomp, dim3(r2, r1, 1), dEb_U, 512, &lrz_time, 0);
    cudaDeviceSynchronize();
    //move zero_U_data back to dU
    thrust::scatter(thrust::device, zero_U_data, zero_U_data + zero_eb_U_count, zero_U_indices, dU_decomp);
    //decompression V
    T* dV_decomp; cudaMalloc(&dV_decomp, r1 * r2 * sizeof(T));
    //move ov_val to dU_decomp accrording to ot_idx
    thrust::scatter(thrust::device, ot_val_V, ot_val_V + *ot_num_V, ot_idx_V, dV_decomp);
    //decompression kernel
    psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, uint16_t>(
        eq_V, /*input*/dV_decomp, /*output*/dV_decomp, dim3(r2, r1, 1), dEb_V, 512, &lrz_time, 0);
    cudaDeviceSynchronize();
    //move zero_V_data back to dV
    thrust::scatter(thrust::device, zero_V_data, zero_V_data + zero_eb_U_count, zero_V_indices, dV_decomp);


    //test compression and decompression
    /*
    cudaStreamCreate(&stream);
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i_count=0;i_count<3;i_count++){
        float ms = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            float temp;
            cudaEventRecord(a, stream);
            psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, uint16_t>(
    	        dU, dim3(r2, r1, 1), eq_U, ot_val_U, ot_idx_U, ot_num_U, dEb_U, 512, &lrz_time, 0);
            //cudaDeviceSynchronize();
            cudaEventRecord(b, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&temp, a, b);
            ms+=temp;
        }
        printf("Compression U elasped time is %f ms, speed GiB/s: %f\n", ms/N, bytes / GiB / (ms / N / 1000));
    }
    cudaStreamDestroy(stream);

    cudaStreamCreate(&stream);
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i_count=0;i_count<3;i_count++){
        float ms = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            float temp;
            cudaEventRecord(a, stream);
            psz::cuhip::GPU_PROTO_c_lorenzo_nd_with_outlier__bypass_outlier_struct__eb_list<T, uint16_t>(
    	        dV, dim3(r2, r1, 1), eq_V, ot_val_V, ot_idx_V, ot_num_V, dEb_V, 512, &lrz_time, 0);
            //cudaDeviceSynchronize();
            cudaEventRecord(b, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&temp, a, b);
            ms+=temp;
        }
        printf("Compression V elasped time is %f ms, speed GiB/s: %f\n", ms/N, bytes / GiB / (ms / N / 1000));
    }
    cudaStreamDestroy(stream);

    cudaStreamCreate(&stream);
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i_count=0;i_count<3;i_count++){
        float ms = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            float temp;
            cudaEventRecord(a, stream);
            psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, uint16_t>(
    	        eq_U, dU_decomp, dU_decomp, dim3(r2, r1, 1), dEb_U, 512, &lrz_time, 0);
            //cudaDeviceSynchronize();
            cudaEventRecord(b, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&temp, a, b);
            ms+=temp;
        }
        printf("Decompression U elasped time is %f ms, speed GiB/s: %f\n", ms/N, bytes / GiB / (ms / N / 1000));
    }
    cudaStreamDestroy(stream);

    cudaStreamCreate(&stream);
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i_count=0;i_count<3;i_count++){
        float ms = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            float temp;
            cudaEventRecord(a, stream);
            psz::cuhip::GPU_PROTO_x_lorenzo_nd__eb_list<T, uint16_t>(
                eq_V, dV_decomp, /dV_decomp, dim3(r2, r1, 1), dEb_V, 512, &lrz_time, 0);
            //cudaDeviceSynchronize();
            cudaEventRecord(b, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&temp, a, b);
            ms+=temp;
        }
        printf("Decompression V elasped time is %f ms, speed GiB/s: %f\n", ms/N, bytes / GiB / (ms / N / 1000));
    }
    cudaStreamDestroy(stream);
    */

    cudaError_t err;
    printf("compute eq done\n");
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(eb_gpu);
        cudaFree(dU);
        cudaFree(dV);
        cudaFree(eq_U);
        cudaFree(eq_V);
        cudaFree(dU_decomp);
        cudaFree(dV_decomp);
        cudaFree(dEb_U);
        cudaFree(dEb_V);
        cudaFree(zero_U_data);
        cudaFree(zero_U_indices);
        cudaFree(zero_V_data);
        cudaFree(zero_V_indices);
        return 0;
    }

    //copy back
    cudaMemcpy(U_decomp,dU_decomp, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_decomp,dV_decomp, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    
    //cudaFree
    cudaFree(eb_gpu);
    cudaFree(dU);
    cudaFree(dV);
    cudaFree(eq_U);
    cudaFree(eq_V);
    cudaFree(dU_decomp);
    cudaFree(dV_decomp);
    cudaFree(dEb_U);
    cudaFree(dEb_V);
    cudaFree(zero_U_data);
    cudaFree(zero_U_indices);
    cudaFree(zero_V_data);
    cudaFree(zero_V_indices);

    return 0;
}

int main(int argc, char ** argv){
    size_t num_elements = 0;
    float * U = readfile<float>(argv[1], num_elements);
    float * V = readfile<float>(argv[2], num_elements);
    int r1 = atoi(argv[3]);
    int r2 = atoi(argv[4]);
    float max_eb = atof(argv[5]);

    size_t result_size = 0;
    //struct timespec start, end;
    //int err = 0;
    //err = clock_gettime(CLOCK_REALTIME, &start);
    cout << "start Compression\n";

    float* ot_val_U; cudaMalloc(&ot_val_U, r2 * r1 * sizeof(float) / 5);
    uint32_t* ot_idx_U; cudaMalloc(&ot_idx_U, r2 * r1 * sizeof(uint32_t) / 5);
    uint32_t* ot_num_U; cudaMallocManaged(&ot_num_U, sizeof(uint32_t));

    float* ot_val_V; cudaMalloc(&ot_val_V, r2 * r1 * sizeof(float) / 5);
    uint32_t* ot_idx_V; cudaMalloc(&ot_idx_V, r2 * r1 * sizeof(uint32_t) / 5);
    uint32_t* ot_num_V; cudaMallocManaged(&ot_num_V, sizeof(uint32_t));

    float * U_decomp = (float *) malloc(r1 * r2 * sizeof(float));
    float * V_decomp = (float *) malloc(r1 * r2 * sizeof(float));

    unsigned char * result = sz_compress_cp_preserve_2d_offline_gpu(U, V, r1, r2, result_size, false, max_eb, ot_val_U, ot_idx_U, ot_num_U, ot_val_V, ot_idx_V, ot_num_V, U_decomp, V_decomp);
    // unsigned char * result = sz_compress_cp_preserve_2d_online_gpu(U, V, r1, r2, result_size, false, max_eb);


    verify(U, U_decomp, num_elements);
    verify(V, V_decomp, num_elements);
    

    writefile((string(argv[1]) + ".out").c_str(), U_decomp, num_elements);
    writefile((string(argv[2]) + ".out").c_str(), V_decomp, num_elements);
    
    free(result);
    free(U);
    free(V);
    free(U_decomp);
    free(V_decomp);
    cudaFree(ot_val_U);
    cudaFree(ot_idx_U);
    cudaFree(ot_num_U);
    cudaFree(ot_val_V);
    cudaFree(ot_idx_V);
    cudaFree(ot_num_V);

}