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
[[nodiscard]] constexpr inline double max_eb_to_keep_position_and_type(const T volatile u0, const T volatile u1, const T volatile u2, const T volatile v0, const T volatile v1, const T volatile v2, 
                                                                    const T x0, const T x1, const T x2, const T y0, const T y1, const T y2){//instant no use for now, future use for online 2024/12/4	
    T u0v1 = u0 * v1;
    T u1v0 = u1 * v0;
    T u0v2 = u0 * v2;
    T u2v0 = u2 * v0;
    T u1v2 = u1 * v2;
    T u2v1 = u2 * v1;
    T det = u0v1 - u1v0 + u1v2 - u2v1 + u2v0 - u0v2;
    T eb = 0;
    if(det != 0){
        bool f1 = (det / (u2v0 - u0v2) >= 1);
        bool f2 = (det / (u1v2 - u2v1) >= 1); 
        bool f3 = (det / (u0v1 - u1v0) >= 1); 
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
[[nodiscard]] constexpr inline double gpu_max_eb_to_keep_position_and_type(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2, 
                                                                    const T x0, const T x1, const T x2, const T y0, const T y1, const T y2){//instant no use for now, future use for online 2024/12/4
    auto gpu_minf = [](auto a, auto b) -> T{ return (a<b)?a:b; };
    auto lambda_gpu_max_eb_to_keep_sign_2 = [](auto u0, auto u1) {
        auto positive = (u0 >= 0 ? u0 : 0) + (u1 >= 0 ? u1 : 0);
        auto negative = (u0 <  0 ? -u0 : 0) + (u1 <  0 ? -u1 : 0);
        auto P =sqrt(positive);
        auto N =sqrt(negative);
        return fabs(P - N) / (P + N);
    };

    auto lambda_gpu_max_eb_to_keep_sign_4 = [](auto u0, auto u1, auto u2, auto u3) {
        auto positive = (u0 >= 0 ? u0 : 0) + (u1 >= 0 ? u1 : 0) + (u2 >= 0 ? u2 : 0) + (u3 >= 0 ? u3 : 0);
        auto negative = (u0 <  0 ? -u0 : 0) + (u1 <  0 ? -u1 : 0) + (u2 <  0 ? -u2 : 0) + (u3 <  0 ? -u3 : 0);
        auto P =sqrt(positive);
        auto N =sqrt(negative);
        return fabs(P - N) / (P + N);
    };

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
        eb = 0;
        if(!f1){
            //T eb_cur = gpu_minf(gpu_max_eb_to_keep_sign_2d_offline_2_degree2(U2V0, -U0V2), gpu_max_eb_to_keep_sign_2d_offline_4_degree2(U0V1, -U1V0, U1V2, -U2V1));
            T eb_cur = gpu_minf(lambda_gpu_max_eb_to_keep_sign_2(U2V0, -U0V2), lambda_gpu_max_eb_to_keep_sign_4(U0V1, -U1V0, U1V2, -U2V1));
            eb = MAX(eb, eb_cur);
        }
        if(!f2){
            //T eb_cur = gpu_minf(gpu_max_eb_to_keep_sign_2d_offline_2_degree2(U1V2, -U2V1), gpu_max_eb_to_keep_sign_2d_offline_4_degree2(U0V1, -U1V0, U2V0, -U0V2));
            T eb_cur = gpu_minf(lambda_gpu_max_eb_to_keep_sign_2(U1V2, -U2V1), lambda_gpu_max_eb_to_keep_sign_4(U0V1, -U1V0, U2V0, -U0V2));
            eb = MAX(eb, eb_cur);
        }
        if(!f3){;
            //T eb_cur = gpu_minf(gpu_max_eb_to_keep_sign_2d_offline_2_degree2(U0V1, -U1V0), gpu_max_eb_to_keep_sign_2d_offline_4_degree2(U1V2, -U2V1, U2V0, -U0V2));
            T eb_cur = gpu_minf(lambda_gpu_max_eb_to_keep_sign_2(U0V1, -U1V0), lambda_gpu_max_eb_to_keep_sign_4(U1V2, -U2V1, U2V0, -U0V2));
            eb = MAX(eb, eb_cur);
        }
        // eb = MINF(eb, DEFAULT_EB);
    }
    return eb;
}

//version 2, enable rectangle blocksize
template <typename T, int TileDim_X = 32, int TileDim_Y = 8>
__global__ void derive_eb_offline_v2(const T* __restrict__ dU, const T* __restrict__ dV, T* __restrict__ dEb, T* __restrict__  dEb_U,  T* __restrict__ dEb_V, int r1, int r2, T max_pwr_eb){
    __shared__ T buf_U[TileDim_Y][TileDim_X+1];
    __shared__ T buf_V[TileDim_Y][TileDim_X+1];
    __shared__ T per_cell_eb_L[TileDim_Y][TileDim_X+1];
    __shared__ T per_cell_eb_U[TileDim_Y][TileDim_X+1];
    __shared__ T buf_eb[TileDim_Y][TileDim_X+1];    
    int row = blockIdx.y * (blockDim.y-2) + threadIdx.y; // global row index
    int col = blockIdx.x * (blockDim.x-2) + threadIdx.x; // global col index
    //int localRow = threadIdx.y; // local row index
    //int localCol = threadIdx.x; // local col index
#define localRow threadIdx.y
#define localCol threadIdx.x

    buf_eb[localRow][localCol] = max_pwr_eb;
    __syncthreads();

    // load data from global memory to shared memory
    if(row < r1 && col < r2){
        buf_U[localRow][localCol] = dU[row * r2 + col];
        buf_V[localRow][localCol] = dV[row * r2 + col];
    }
    __syncthreads();
    
    //bottleneck is here
    if(localRow<TileDim_Y-1 && localCol<TileDim_X-1){
        per_cell_eb_U[localRow][localCol] = gpu_max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow][localCol+1], buf_U[localRow+1][localCol+1],
            buf_V[localRow][localCol], buf_V[localRow][localCol+1], buf_V[localRow+1][localCol+1], static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0));
        per_cell_eb_L[localRow][localCol] = gpu_max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow+1][localCol], buf_U[localRow+1][localCol+1],
            buf_V[localRow][localCol], buf_V[localRow+1][localCol], buf_V[localRow+1][localCol+1], static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0));
    }
    __syncthreads();

    /************************************记得验证对错时要删除*******************************************************/
    return;

    /*
    //printf buf_U
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 3 && blockIdx.y == 3) {
        printf("Here is block(%d,%d)\n", blockIdx.y, blockIdx.x);
        for (int i = 0; i < TileDim; i++) {
            for (int j = 0; j < TileDim ; j++) {
                printf("%f ", buf_U[i][j]); // 假设 buf_U 的类型为 float
            }
            printf("\n"); // 打印每一行后换行
        }

    }
    __syncthreads();
    */
    /*
    //printf buf_eb
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("Here is block(%d,%d)\n", blockIdx.y, blockIdx.x);
        for (int i = 0; i < TileDim-2; i++) {
            for (int j = 0; j < TileDim-2 ; j++) {
                printf("%f ", buf_eb[i][j]); // 假设 buf_U 的类型为 float
            }
            printf("\n"); // 打印每一行后换行
        }
    }
    __syncthreads();
    */ 
    
    T localmin;
    if(localRow<TileDim_Y-2 && localCol<TileDim_X-2)
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
    __syncthreads();

    /*
    //For centeral part bug_check
    if (localRow<TileDim_Y-1 && localCol<TileDim_X-1 && row*r2+col == 24508) {
        buf_eb[localRow][localCol] = min(per_cell_eb_U[localRow][localCol], per_cell_eb_L[localRow][localCol]);
        printf("buf_eb[%d][%d]: %.4f\n", localRow, localCol, buf_eb[localRow][localCol]);

        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_U[localRow + 1][localCol]);
        printf("buf_eb[%d][%d] after U[%d+1][%d]: %.4f\n", localRow, localCol, localRow, localCol, buf_eb[localRow][localCol]);

        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_L[localRow][localCol + 1]);
        printf("buf_eb[%d][%d] after L[%d][%d+1]: %.4f\n", localRow, localCol, localRow, localCol, buf_eb[localRow][localCol]);

        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_U[localRow + 1][localCol + 1]);
        printf("buf_eb[%d][%d] after U[%d+1][%d+1]: %.4f\n", localRow, localCol, localRow, localCol, buf_eb[localRow][localCol]);

        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_L[localRow + 1][localCol + 1]);
        printf("buf_eb[%d][%d] after L[%d+1][%d+1]: %.4f\n", localRow, localCol, localRow, localCol, buf_eb[localRow][localCol]);

        // 打印每个相关变量的值
        printf("Variables:\n");
        printf("row: %d, col: %d\n", row, col);
        printf("localRow: %d, localCol: %d\n", localRow, localCol);
        printf("per_cell_eb_U[%d][%d]: %.4f\n", localRow, localCol, per_cell_eb_U[localRow][localCol]);
        printf("per_cell_eb_L[%d][%d]: %.4f\n", localRow, localCol, per_cell_eb_L[localRow][localCol]);
        printf("per_cell_eb_U[%d+1][%d]: %.4f\n", localRow, localCol, per_cell_eb_U[localRow + 1][localCol]);
        printf("per_cell_eb_L[%d][%d+1]: %.4f\n", localRow, localCol, per_cell_eb_L[localRow][localCol + 1]);
        printf("per_cell_eb_U[%d+1][%d+1]: %.4f\n", localRow, localCol, per_cell_eb_U[localRow + 1][localCol + 1]);
        printf("per_cell_eb_L[%d+1][%d+1]: %.4f\n", localRow, localCol, per_cell_eb_L[localRow + 1][localCol + 1]);
    }
    __syncthreads();
    */

    if(row<r1-2 && col<r2-2 && localRow<TileDim_Y-2 && localCol<TileDim_X-2){
        auto temp = localmin * fabs(buf_U[localRow+1][localCol+1]);
        temp = (temp < std::numeric_limits<T>::epsilon() ?  0 : temp);
	    int id = log2(temp / std::numeric_limits<T>::epsilon())/2.0;
	    temp = pow(4, id) * std::numeric_limits<T>::epsilon();
        dEb_U[(row+1) * r2 + (col+1)] = temp;
        
        temp = localmin * fabs(buf_V[localRow+1][localCol+1]);
        temp = (temp < std::numeric_limits<T>::epsilon() ?  0 : temp);
	    id = log2(temp / std::numeric_limits<T>::epsilon())/2.0;
	    temp = pow(4, id) * std::numeric_limits<T>::epsilon();
        dEb_V[(row+1) * r2 + (col+1)] =  temp;
    }
    __syncthreads();
    
    if(row == 0 || col ==0 || row==r1-1 || col == r2-1){
        dEb_U[row * r2 + col] = 0;
        dEb_V[row * r2 + col] = 0;
    }
    __syncthreads();
    /*
    //Edge cases
    //top edge
    if(row==0 && col < r2-2 && localRow<TileDim_Y-2 && localCol<TileDim_X-2){
        buf_eb[0][localCol] = max_pwr_eb;
        buf_eb[0][localCol] = min(per_cell_eb_U[localRow][localCol], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_U[localRow][localCol+1], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_L[localRow][localCol+1], buf_eb[0][localCol]);
        dEb_U[col+1] = buf_eb[0][localCol] * buf_U[0][localCol+1];
        dEb_V[col+1] = buf_eb[0][localCol] * buf_V[0][localCol+1];
    }
    __syncthreads();

    //bottom edge
    if(row==r1-1 && col < r2-2 && localRow<TileDim_Y-2 && localCol<TileDim_X-2){
        buf_eb[0][localCol] = max_pwr_eb;
        buf_eb[0][localCol] = min(per_cell_eb_U[localRow-1][localCol], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_L[localRow-1][localCol], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_L[localRow-1][localCol+1], buf_eb[0][localCol]);
        dEb_U[row*r2 + col+1] = buf_eb[0][localCol] * buf_U[localRow][localCol+1];
        dEb_V[row*r2 + col+1] = buf_eb[0][localCol] * buf_V[localRow][localCol+1];
    }
    __syncthreads();

    //left edge
    if(col==0 && row < r1-2 && localRow<TileDim_Y-2 && localCol<TileDim_X-2){
        buf_eb[localRow][0] = max_pwr_eb;
        buf_eb[localRow][0] = min(per_cell_eb_L[localRow][localCol], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_U[localRow+1][localCol], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_L[localRow+1][localCol], buf_eb[localRow][0]);
        dEb_U[(row+1)*r2] = buf_eb[localRow][0] * buf_U[localRow+1][0];
        dEb_V[(row+1)*r2] = buf_eb[localRow][0] * buf_V[localRow+1][0];
    }
   __syncthreads();

    //right edge
    if(row < r1-2 && col==r2-1 && localRow<TileDim_Y-2 && localCol<TileDim_X-2){
        buf_eb[localRow][0] = max_pwr_eb;
        buf_eb[localRow][0] = min(per_cell_eb_L[localRow][localCol-1], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_U[localRow][localCol-1], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_U[localRow+1][localCol-1], buf_eb[localRow][0]);
        dEb_U[(row+1)*r2 + r2-1] = buf_eb[localRow][0] * buf_U[localRow+1][localCol];
        dEb_V[(row+1)*r2 + r2-1] = buf_eb[localRow][0] * buf_V[localRow+1][localCol];
    }
    __syncthreads();

    //top left corner
    if(row==0&&col==0){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_U[0][0], buf_eb[0][0]);
        buf_eb[0][0] = min(per_cell_eb_L[0][0], buf_eb[0][0]);
        dEb_U[0] = buf_eb[0][0]*buf_U[0][0];
        dEb_V[0] = buf_eb[0][0]*buf_V[0][0];
    }
    
    //top right corner
    if(row==0&&col==r2-1){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_U[0][localCol-1], buf_eb[0][0]);
        dEb_U[col] = buf_eb[0][0]*buf_U[0][localCol];
        dEb_V[col] = buf_eb[0][0]*buf_V[0][localCol];
    }

    //bottom left corner
    if(row==r1-1&&col==0){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_L[localRow-1][0], buf_eb[0][0]);
        dEb_U[row*r2] = buf_eb[0][0]*buf_U[localRow][0];
        dEb_V[row*r2] = buf_eb[0][0]*buf_V[localRow][0];
    }

    //bottom right corner
    if(row==r1-1&&col==r2-1){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_U[localRow-1][localCol-1], buf_eb[0][0]);
        buf_eb[0][0] = min(per_cell_eb_L[localRow-1][localCol-1], buf_eb[0][0]);
        dEb_U[row*r2 + col] = buf_eb[0][0]*buf_U[localRow][localCol];
        dEb_V[row*r2 + col] = buf_eb[0][0]*buf_V[localRow][localCol];
    } 
    __syncthreads();
    */
}

// compression with pre-computed error bounds
template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_offline_gpu(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, T max_pwr_eb, 
        T* ot_val_U, uint32_t* ot_idx_U, uint32_t* ot_num_U, T* ot_val_V, uint32_t* ot_idx_V, uint32_t* ot_num_V, T* U_decomp, T* V_decomp){
    
    size_t num_elements = r1 * r2;
    T * eb = (T *) malloc(num_elements * sizeof(T));
    for(int i=0; i<num_elements; i++) eb[i] = max_pwr_eb;
    T * eb_gpu; cudaMalloc(&eb_gpu, r1 * r2 * sizeof(T));
    cudaMemset(eb_gpu, max_pwr_eb, r2 * r1 * sizeof(T));
    
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
                	V_row_pos[offset[0]], V_row_pos[offset[1]], V_row_pos[offset[2]], X[0][1], X[1][1], X[2][1],
                	X[0][0], X[1][0], X[2][0]);
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
    cudaError_t err;

    T *dU, *dV, *dEb_U, *dEb_V;
    cudaMalloc(&dU, r1 * r2 * sizeof(T));
    cudaMalloc(&dV, r1 * r2 * sizeof(T));
    cudaMalloc(&dEb_U, r1 * r2 * sizeof(T));
    cudaMalloc(&dEb_V, r1 * r2 * sizeof(T));
    cudaMemcpy(dU, U, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    


    cudaStream_t stream;
    cudaEvent_t a, b;

    //run Kernel v2:
    dim3 blockSize_v2(32, 8, 1);
    dim3 gridSize_v2((r2 + (blockSize_v2.x-2) - 1) / (blockSize_v2.x-2), (r1 + (blockSize_v2.y-2) - 1) / (blockSize_v2.y-2));
    printf("gridSize_v2: %d, %d\n", gridSize_v2.x, gridSize_v2.y);
    derive_eb_offline_v2<<<gridSize_v2, blockSize_v2>>>(dU, dV, eb_gpu, dEb_U, dEb_V, r1, r2, max_pwr_eb);
    cudaDeviceSynchronize();
    printf("compute V2 eb_gpu done\n"); //
    //printf("speed GiB/s: %f\n", bytes / GiB / (ms / 1000));
    
    auto bytes = 3600 * 2400 * 4 * 2.0;
    auto GiB = 1024 * 1024 * 1024.0;
    int N = 30;
    cudaStreamCreate(&stream);
    cudaEventCreate(&a), cudaEventCreate(&b);
    for (int i_count=0;i_count<3;i_count++){
        float ms = 0.0;
        for (size_t i = 0; i < N; i++)
        {
            float temp;
            cudaEventRecord(a, stream);
            derive_eb_offline_v2<<<gridSize_v2, blockSize_v2, 0, stream>>>(dU, dV, eb_gpu, dEb_U, dEb_V, r1, r2, max_pwr_eb);
            //cudaDeviceSynchronize();
            cudaEventRecord(b, stream);
            cudaStreamSynchronize(stream);
            cudaEventElapsedTime(&temp, a, b);
            ms+=temp;
        }
        printf("elasped time is %f ms, speed GiB/s: %f\n", ms/N, bytes / GiB / (ms / N / 1000));
    }
    cudaStreamDestroy(stream);

    //verify derive_eb
    
    //cpu eb_u, eb_v
    const int base = 4;
	T log2_of_base = log2(base);
	const T threshold = std::numeric_limits<T>::epsilon();
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
            diff = fabs(eb_u_gpu[i] - eb_u[i]);
            if(diff > maxdiff)
            { 
                maxdiff = diff;
                maxdiff_index = i;    
            }
            if (diff > std::numeric_limits<float>::epsilon()) {
                //printf("error. eb_u_gpu: %5.2f, eb_u: %5.2f,%d\n", eb_u_gpu[i],eb_u[i],i);
                //break;
                count++;
            }
        }
    }
    printf("maxdiff: %f, maxdiff_index: %d, error count: %d\n", maxdiff, maxdiff_index, count);
    printf("eb_u_gpu: %f, eb_u: %f\n", eb_u_gpu[maxdiff_index], eb_u[maxdiff_index]);

    diff = 0.0;
    maxdiff = 0.0;
    count=0;
    maxdiff_index = 0;
    for (int i = 1; i < r1-1; i++){
        for(int j = 1; j < r2-1; j++){
            diff = fabs(eb_v_gpu[i] - eb_v[i]);
            if(diff > maxdiff)
            { 
                maxdiff = diff;
                maxdiff_index = i;    
            }
            if (diff > std::numeric_limits<float>::epsilon()) {
                //printf("error. eb_u_gpu: %5.2f, eb_u: %5.2f,%d\n", eb_u_gpu[i],eb_u[i],i);
                //break;
                count++;
            }
        }
    }
    printf("maxdiff: %f, maxdiff_index: %d, error count: %d\n", maxdiff, maxdiff_index, count);
    printf("eb_v_gpu: %f, eb_v: %f\n", eb_v_gpu[maxdiff_index], eb_v[maxdiff_index]);
    

    // deal with eb =zero  
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
    //cudaMemcpy(eb_gpu, dEb, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(U_decomp,dU_decomp, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(V_decomp,dV_decomp, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);

    //Compute time:
    //GPU time 
    /*
    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    size_t N=1;
    */
    //V1 GPU
    /*
    for (int i_count=0;i_count<10;i_count++){
        cudaEventRecord(beg);
        for (size_t i = 0; i < N; i++)
        {
            derive_eb_offline<<<gridSize, blockSize>>>(dU, dV, dEb, r1, r2, max_pwr_eb);
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;
        printf("Average GPU_V1 elasped time: %f second\n", elapsed_time/N);
    }
    fflush(stdout);
    */
    //V2 GPU
    /*
    for (int i_count=0;i_count<10;i_count++){
        auto start = std::chrono::high_resolution_clock::now();
        derive_eb_offline_v2<<<gridSize_v2, blockSize_v2>>>(dU, dV, eb_gpu, r1, r2, max_pwr_eb);
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = stop - start;

        printf("derive_eb time is %lf, throughput is %lf GiBs\n", elapsed.count()/1000, (4*r1*r2*2.0/(elapsed.count()/1000))/(1024*1024*1024));
    }
    fflush(stdout);
    */
    //CPU time
    /*
    double t0,t1;
    for (int i_count=0;i_count<10;i_count++){
        t0=get_sec();
        for (size_t i = 0; i < N; i++)
        {
            U_pos = U;
            V_pos = V;
            eb_pos = eb;
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
                        T max_cur_eb = gpu_max_eb_to_keep_position_and_type(U_row_pos[offset[0]], U_row_pos[offset[1]], U_row_pos[offset[2]],
                        	V_row_pos[offset[0]], V_row_pos[offset[1]], V_row_pos[offset[2]], X[0][1], X[1][1], X[2][1],
                        	X[0][0], X[1][0], X[2][0]);
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
        }
        t1=get_sec();
        printf("Average CPU elasped time: %f second\n", (t1-t0)/N);
    }
    */
    
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

    //compression
    /*
    double * eb_u = (double *) malloc(num_elements * sizeof(double));
    double * eb_v = (double *) malloc(num_elements * sizeof(double));
    int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
    int * eb_quant_index_pos = eb_quant_index;
    const int base = 4;
    double log2_of_base = log2(base);
    const double threshold = std::numeric_limits<double>::epsilon();
    for(int i=0; i<num_elements; i++){
    	eb_u[i] = fabs(U[i]) * eb[i];
    	*(eb_quant_index_pos ++) = eb_exponential_quantize(eb_u[i], base, log2_of_base, threshold);
    	// *(eb_quant_index_pos ++) = eb_linear_quantize(eb_u[i], 1e-2);
    	if(eb_u[i] < threshold) eb_u[i] = 0;
    }
    for(int i=0; i<num_elements; i++){
    	eb_v[i] = fabs(V[i]) * eb[i];
    	*(eb_quant_index_pos ++) = eb_exponential_quantize(eb_v[i], base, log2_of_base, threshold);
    	// *(eb_quant_index_pos ++) = eb_linear_quantize(eb_v[i], 1e-2);
    	if(eb_v[i] < threshold) eb_v[i] = 0;
    }
    free(eb);
    printf("quantize eb done\n");
    unsigned char * compressed_eb = (unsigned char *) malloc(2*num_elements*sizeof(int));
    unsigned char * compressed_eb_pos = compressed_eb; 
    Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_eb_pos);
    size_t compressed_eb_size = compressed_eb_pos - compressed_eb;
    size_t compressed_u_size = 0;
    size_t compressed_v_size = 0;
    unsigned char * compressed_u = sz_compress_2d_with_eb(U, eb_u, r1, r2, compressed_u_size);
    unsigned char * compressed_v = sz_compress_2d_with_eb(V, eb_v, r1, r2, compressed_v_size);
    printf("eb_size = %ld, u_size = %ld, v_size = %ld\n", compressed_eb_size, compressed_u_size, compressed_v_size);
    free(eb_u);
    free(eb_v);
    compressed_size = sizeof(int) + sizeof(size_t) + compressed_eb_size + sizeof(size_t) + compressed_u_size + sizeof(size_t) + compressed_v_size;
    unsigned char * compressed = (unsigned char *) malloc(compressed_size);
    unsigned char * compressed_pos = compressed;
    write_variable_to_dst(compressed_pos, base);
    write_variable_to_dst(compressed_pos, threshold);
    write_variable_to_dst(compressed_pos, compressed_eb_size);
    write_variable_to_dst(compressed_pos, compressed_u_size);
    write_variable_to_dst(compressed_pos, compressed_v_size);
    write_array_to_dst(compressed_pos, compressed_eb, compressed_eb_size);
    write_array_to_dst(compressed_pos, compressed_u, compressed_u_size);
    printf("compressed_pos = %ld\n", compressed_pos - compressed);
    write_array_to_dst(compressed_pos, compressed_v, compressed_v_size);
    free(compressed_eb);
    free(compressed_u);
    free(compressed_v);
    */

    //free(eb);
    return 0;
}

//online fuction
/* 
template<typename T>
constexpr inline void get_adjugate_matrix_for_position(const T x0, const T x1, const T x2, const T y0, const T y1, const T y2, T c[4]){
  T determinant = (x0 - x2)*(y1 - y2) - (x1 - x2)*(y0 - y2);
  c[0] = (y1 - y2) / determinant;
  c[1] = -(y0 - y2) / determinant;
  c[2] = -(x1 - x2) / determinant;
  c[3] = (x0 - x2) / determinant;
  // printf("%.4g, %.2g %.2g %.2g %.2g\n", determinant, c[0], c[1], c[2], c[3]);
  // exit(0);
}

// maximal error bound to keep the sign of A*(1 + e_1) + B*(1 + e_2) + C
template<typename T>
[[nodiscard]] constexpr static inline double max_eb_to_keep_sign_2d_online(const T A, const T B, const T C=0){
	double fabs_sum = (fabs(A) + fabs(B));
	if(fabs_sum == 0) return 0;
	return fabs(A + B + C) / fabs_sum;
}

template<typename T>
double 
derive_cp_eb_for_positions_online(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2, const T c[4]){//, conditions_2d& cond){
	// if(!cond.computed){
	//     double M0 = u2*v0 - u0*v2;
	//     double M1 = u1*v2 - u2*v1;
	//     double M2 = u0*v1 - u1*v0;
	//     double M = M0 + M1 + M2;
	//     cond.singular = (M == 0);
	//     if(cond.singular) return 0;
	//     cond.flags[0] = (M0 == 0) || (M / M0 >= 1);
	//     cond.flags[1] = (M1 == 0) || (M / M1 >= 1);
	//     cond.flags[2] = (M2 == 0) || (M / M2 >= 1);
	//     cond.computed = true;
	// }
	// else{
	//     if(cond.singular) return 0;
	// }
	// const bool * flag = cond.flags;
	// bool f1 = flag[0];
	// bool f2 = flag[1]; 
	// bool f3 = flag[2];
	double M0 = u2*v0 - u0*v2;
    double M1 = u1*v2 - u2*v1;
    double M2 = u0*v1 - u1*v0;
    double M = M0 + M1 + M2;
	if(M == 0) return 0;
	bool f1 = (M0 == 0) || (M / M0 >= 1);
	bool f2 = (M1 == 0) || (M / M1 >= 1); 
	bool f3 = (M2 == 0) || (M / M2 >= 1);
	double eb = 0;
	if(f1 && f2 && f3){
		// eb = max_eb_to_keep_position_online(u0v1, u1v0, u1v2, u2v1, u2v0, u0v2);
		eb = 0;
	}
	else{
		eb = 0;
		if(!f1){
			// W1(W0 + W2)
			double cur_eb = MINF(max_eb_to_keep_sign_2d_online(u2*v0, -u0*v2), max_eb_to_keep_sign_2d_online(-u2*v1, u1*v2, u0*v1 - u1*v0));
			eb = MAX(eb, cur_eb);
		}
		if(!f2){
			// W0(W1 + W2)
			double cur_eb = MINF(max_eb_to_keep_sign_2d_online(-u2*v1, u1*v2), max_eb_to_keep_sign_2d_online(u2*v0, -u0*v2, u0*v1 - u1*v0));
			eb = MAX(eb, cur_eb);				
		}
		if(!f3){
			// W2(W0 + W1)
			double cur_eb = max_eb_to_keep_sign_2d_online(u2*v0 - u2*v1, u1*v2 - u0*v2);
			eb = MAX(eb, cur_eb);				
		}
	}
	return eb;
}

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_online_gpu(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb){
    size_t num_elements = r1 * r2;
    T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
    memcpy(decompressed_U, U, num_elements*sizeof(T));
    T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
    memcpy(decompressed_V, V, num_elements*sizeof(T));
    int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
    int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
    int * eb_quant_index_pos = eb_quant_index;
    int * data_quant_index_pos = data_quant_index;
    // next, row by row
    const int base = 4;
    const double log_of_base = log2(base);
    const int capacity = 65536;
    const int intv_radius = (capacity >> 1);
    unpred_vec<T> unpred_data;
    T * U_pos = decompressed_U;
    T * V_pos = decompressed_V;
    // offsets to get six adjacent triangle indices
    // the 7-th rolls back to T0
    
    //      T3	T4
    //	T2	X 	T5
    //	T1	T0 (T6)

    const int offsets[7] = {
    	-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
    };
    const T x[6][3] = {
    	{1, 0, 1},
    	{0, 0, 1},
    	{0, 1, 1},
    	{0, 1, 0},
    	{1, 1, 0},
    	{1, 0, 0}
    };
    const T y[6][3] = {
    	{0, 0, 1},
    	{0, 1, 1},
    	{0, 1, 0},
    	{1, 1, 0},
    	{1, 0, 0},
    	{1, 0, 1}
    };
    T inv_C[6][4];
    for(int i=0; i<6; i++){
    	get_adjugate_matrix_for_position(x[i][0], x[i][1], x[i][2], y[i][0], y[i][1], y[i][2], inv_C[i]);
    }
    int index_offset[6][2][2];
    for(int i=0; i<6; i++){
    	for(int j=0; j<2; j++){
    		index_offset[i][j][0] = x[i][j] - x[i][2];
    		index_offset[i][j][1] = y[i][j] - y[i][2];
    	}
    }
    double threshold = std::numeric_limits<double>::epsilon();
    // conditions_2d cond;
    for(int i=0; i<r1; i++){
    	// printf("start %d row\n", i);
    	T * cur_U_pos = U_pos;
    	T * cur_V_pos = V_pos;
    	for(int j=0; j<r2; j++){
    		double required_eb = max_pwr_eb;
    		// derive eb given six adjacent triangles
    		for(int k=0; k<6; k++){
    			bool in_mesh = true;
    			for(int p=0; p<2; p++){
    				// reserved order!
    				if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
    					in_mesh = false;
    					break;
    				}
    			}
    			if(in_mesh){
    				required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
    					cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0], inv_C[k]));
    			}
    		}
    		if(required_eb > 0){
    			bool unpred_flag = false;
    			T decompressed[2];
    			// compress U and V
    			for(int k=0; k<2; k++){
    				T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
    				T cur_data = *cur_data_pos;
    				double abs_eb = fabs(cur_data) * required_eb;
    				eb_quant_index_pos[k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
    				// eb_quant_index_pos[k] = eb_linear_quantize(abs_eb, 1e-3);
    				if(eb_quant_index_pos[k] > 0){
    					// get adjacent data and perform Lorenzo
    					
    					//	d2 X
    					//	d0 d1l
    					
    					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
    					T d1 = (i) ? cur_data_pos[-r2] : 0;
    					T d2 = (j) ? cur_data_pos[-1] : 0;
    					T pred = d1 + d2 - d0;
    					double diff = cur_data - pred;
    					double quant_diff = fabs(diff) / abs_eb + 1;
    					if(quant_diff < capacity){
    						quant_diff = (diff > 0) ? quant_diff : -quant_diff;
    						int quant_index = (int)(quant_diff/2) + intv_radius;
    						data_quant_index_pos[k] = quant_index;
    						decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
    						// check original data
    						if(fabs(decompressed[k] - cur_data) >= abs_eb){
    							unpred_flag = true;
    							break;
    						}
    					}
    					else{
    						unpred_flag = true;
    						break;
    					}
    				}
    				else unpred_flag = true;
    			}
    			if(unpred_flag){
    				// recover quant index
    				*(eb_quant_index_pos ++) = 0;
    				*(eb_quant_index_pos ++) = 0;
    				*(data_quant_index_pos ++) = intv_radius;
    				*(data_quant_index_pos ++) = intv_radius;
    				unpred_data.push_back(*cur_U_pos);
    				unpred_data.push_back(*cur_V_pos);
    			}
    			else{
    				eb_quant_index_pos += 2;
    				data_quant_index_pos += 2;
    				// assign decompressed data
    				*cur_U_pos = decompressed[0];
    				*cur_V_pos = decompressed[1];
    			}
    		}
    		else{
    			// record as unpredictable data
    			*(eb_quant_index_pos ++) = 0;
    			*(eb_quant_index_pos ++) = 0;
    			*(data_quant_index_pos ++) = intv_radius;
    			*(data_quant_index_pos ++) = intv_radius;
    			unpred_data.push_back(*cur_U_pos);
    			unpred_data.push_back(*cur_V_pos);
    		}
    		cur_U_pos ++, cur_V_pos ++;
    	}
    	U_pos += r2;
    	V_pos += r2;
    }
    free(decompressed_U);
    free(decompressed_V);
    printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
    
    //unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
    //unsigned char * compressed_pos = compressed;
    //write_variable_to_dst(compressed_pos, base);
    //write_variable_to_dst(compressed_pos, threshold);
    //write_variable_to_dst(compressed_pos, intv_radius);
    //size_t unpredictable_count = unpred_data.size();
    //write_variable_to_dst(compressed_pos, unpredictable_count);
    //write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
    //Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
    //free(eb_quant_index);
    //Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
    //printf("pos = %ld\n", compressed_pos - compressed);
    //free(data_quant_index);
    //compressed_size = compressed_pos - compressed;
    //return compressed;	
    
    return 0;
}
*/

int main(int argc, char ** argv){
    size_t num_elements = 0;
    float * U = readfile<float>(argv[1], num_elements);
    float * V = readfile<float>(argv[2], num_elements);
    int r1 = atoi(argv[3]);
    int r2 = atoi(argv[4]);
    float max_eb = atof(argv[5]);

    /*
    // Remember to delete when real using
    // Test Use for U and V
    // Initialize U and V with sequential values
    for (int i = 0; i < r1; ++i) {
        for (int j = 0; j < r2; ++j) {
            U[i * r2 + j] = i * r2 + j; // Row-major order
            V[i * r2 + j] = i * r2 + j; // Or any other initialization
        }
    }
    */

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
    // unsigned char * result = sz_compress_cp_preserve_2d_offline_log(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz_compress_cp_preserve_2d_online_gpu(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz_compress_cp_preserve_2d_online_abs(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz_compress_cp_preserve_2d_online_abs_relax_FN(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz_compress_cp_preserve_2d_bilinear_online_log(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz_compress_cp_preserve_2d_online_log(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz3_compress_cp_preserve_2d_online_abs(U, V, r1, r2, result_size, false, max_eb);

    /*
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
    err = clock_gettime(CLOCK_REALTIME, &end);
    cout << "Compression time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    cout << "Compressed size = " << lossless_outsize << ", ratio = " << (2*num_elements*sizeof(float)) * 1.0/lossless_outsize << endl;
    free(result);
    // exit(0);
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
    float * dec_U = NULL;
    float * dec_V = NULL;

    sz_decompress_cp_preserve_2d_offline<float>(result, r1, r2, dec_U, dec_V);
    // sz_decompress_cp_preserve_2d_offline_log<float>(result, r1, r2, dec_U, dec_V);
    // sz_decompress_cp_preserve_2d_online<float>(result, r1, r2, dec_U, dec_V);
    // sz_decompress_cp_preserve_2d_online_log<float>(result, r1, r2, dec_U, dec_V);
    // sz3_decompress_cp_preserve_2d_online<float>(result, r1, r2, dec_U, dec_V);
    err = clock_gettime(CLOCK_REALTIME, &end);
    cout << "Decompression time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    free(result_after_lossless);
    */
    verify(U, U_decomp, num_elements);
    verify(V, V_decomp, num_elements);
    

    writefile((string(argv[1]) + ".out").c_str(), U_decomp, num_elements);
    writefile((string(argv[2]) + ".out").c_str(), V_decomp, num_elements);
    
    free(result);
    free(U);
    free(V);
    cudaFree(ot_val_U);
    cudaFree(ot_idx_U);
    cudaFree(ot_num_U);
    cudaFree(ot_val_V);
    cudaFree(ot_idx_V);
    cudaFree(ot_num_V);
    
    //free(dec_U);
    //free(dec_V);
}