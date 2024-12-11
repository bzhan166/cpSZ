#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_3d.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include "sz3_utils.hpp"

#include "sz_lossless.hpp"
#include "utils.hpp"
using namespace std;

template<typename T>
[[nodiscard]] constexpr inline T max_eb_to_keep_sign_2d_offline_2(const T volatile u0, const T volatile u1, const int degree=2){
    T positive = 0;
    T negative = 0;
    accumulate(u0, positive, negative);
    accumulate(u1, positive, negative);
    return max_eb_to_keep_sign(positive, negative, degree);
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

template <typename T, int TileDim = 16>
__global__ void derive_eb(const T* dU, const T* dV, T* dEb, int r1, int r2, T max_pwr_eb){
    __shared__ T buf_U[TileDim][TileDim];
    __shared__ T buf_V[TileDim][TileDim];
    __shared__ T per_cell_eb_L[TileDim-1][TileDim-1];
    __shared__ T per_cell_eb_U[TileDim-1][TileDim-1];
    __shared__ T buf_eb[TileDim-2][TileDim-2];    
    int row = blockIdx.y * (blockDim.y-2) + threadIdx.y; // global row index
    int col = blockIdx.x * (blockDim.x-2) + threadIdx.x; // global col index
    int localRow = threadIdx.y; // local row index
    int localCol = threadIdx.x; // local col index
/*  
    //initial shared memory
    buf_U[localRow][localCol] = 0;//std::numeric_limits<T>::max();
    buf_V[localRow][localCol] = 0;//std::numeric_limits<T>::max();
*/
    buf_eb[localRow][localCol] = max_pwr_eb;
    __syncthreads();

    // load data from global memory to shared memory
    if(row < r1 && col < r2){
        buf_U[localRow][localCol] = dU[row * r2 + col];
        buf_V[localRow][localCol] = dV[row * r2 + col];
    }
    __syncthreads();

    if(localRow<TileDim-1 && localCol<TileDim-1){
        per_cell_eb_U[localRow][localCol] = max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow][localCol+1], buf_U[localRow+1][localCol+1],
            buf_V[localRow][localCol], buf_V[localRow][localCol+1], buf_V[localRow+1][localCol+1], static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0));
        per_cell_eb_L[localRow][localCol] = max_eb_to_keep_position_and_type(buf_U[localRow][localCol], buf_U[localRow+1][localCol], buf_U[localRow+1][localCol+1],
            buf_V[localRow][localCol], buf_V[localRow+1][localCol], buf_V[localRow+1][localCol+1], static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0),  static_cast<T>(0));
    }
    __syncthreads();


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

    if(row<r1 && col<r2 && localRow<TileDim-2 && localCol<TileDim-2){
        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_U[localRow][localCol]);
        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_L[localRow][localCol]);
        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_U[localRow+1][localCol]);
        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_L[localRow][localCol+1]);
        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_U[localRow+1][localCol+1]);
        buf_eb[localRow][localCol] = min(buf_eb[localRow][localCol], per_cell_eb_L[localRow+1][localCol+1]);
    }
    __syncthreads();

/*
    //For centeral part bug_check
    if (row*r2+col == 7199) {
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

    if(row<r1-2 && col<r2-2 && localRow<TileDim-2 && localCol<TileDim-2){
        dEb[(row+1) * r2 + (col+1)] = buf_eb[localRow][localCol];
    }
    __syncthreads();

    //TODO Edge cases
    //top edge
    if(row==0 && col < r2-2 && localRow<TileDim-2 && localCol<TileDim-2){
        buf_eb[0][localCol] = max_pwr_eb;
        buf_eb[0][localCol] = min(per_cell_eb_U[localRow][localCol], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_U[localRow][localCol+1], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_L[localRow][localCol+1], buf_eb[0][localCol]);
        dEb[row*r2 + col+1] = buf_eb[0][localCol];
    }
    __syncthreads();

    //bottom edge
    if(row==r1-1 && col < r2-2 && localRow<TileDim-2 && localCol<TileDim-2){
        buf_eb[0][localCol] = max_pwr_eb;
        buf_eb[0][localCol] = min(per_cell_eb_U[localRow-1][localCol], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_L[localRow-1][localCol], buf_eb[0][localCol]);
        buf_eb[0][localCol] = min(per_cell_eb_L[localRow-1][localCol+1], buf_eb[0][localCol]);
        dEb[row*r2 + col+1] = buf_eb[0][localCol];
    }
    __syncthreads();

    //left edge
    if(col==0 && row < r1-2 && localRow<TileDim-2 && localCol<TileDim-2){
        buf_eb[localRow][0] = max_pwr_eb;
        buf_eb[localRow][0] = min(per_cell_eb_L[localRow][localCol], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_U[localRow+1][localCol], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_L[localRow+1][localCol], buf_eb[localRow][0]);
        dEb[(row+1)*r2 + col] = buf_eb[localRow][0];
    }
    __syncthreads();

    //right edge
    if(row < r1-2 && col==r2-1 && localRow<TileDim-2 && localCol<TileDim-2){
        buf_eb[localRow][0] = max_pwr_eb;
        buf_eb[localRow][0] = min(per_cell_eb_L[localRow][localCol-1], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_U[localRow][localCol-1], buf_eb[localRow][0]);
        buf_eb[localRow][0] = min(per_cell_eb_U[localRow+1][localCol-1], buf_eb[localRow][0]);
        dEb[(row+1)*r2 + col] = buf_eb[localRow][0];
    }
    __syncthreads();

    //top left corner
    if(row==0&&col==0){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_U[0][0], buf_eb[0][0]);
        buf_eb[0][0] = min(per_cell_eb_L[0][0], buf_eb[0][0]);
        dEb[0] = buf_eb[0][0];
    }

    //top right corner
    if(row==0&&col==r2-1){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_U[0][localCol-1], buf_eb[0][0]);
        dEb[r2-1] = buf_eb[0][0];
    }

    //bottom left corner
    if(row==r1-1&&col==0){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_L[localRow-1][0], buf_eb[0][0]);
        dEb[(r1-1)*r2] = buf_eb[0][0];
    }

    //bottom right corner
    if(row==r1-1&&col==r2-1){
        buf_eb[0][0] = max_pwr_eb;
        buf_eb[0][0] = min(per_cell_eb_U[localRow-1][localCol-1], buf_eb[0][0]);
        buf_eb[0][0] = min(per_cell_eb_L[localRow-1][localCol-1], buf_eb[0][0]);
        dEb[(r1-1)*r2+r2-1] = buf_eb[0][0];
    }
    __syncthreads();

}

// compression with pre-computed error bounds
template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_offline_cpu_gpu(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, T max_pwr_eb){

    size_t num_elements = r1 * r2;
    T * eb = (T *) malloc(num_elements * sizeof(T));
    for(int i=0; i<num_elements; i++) eb[i] = max_pwr_eb;
    T * eb_gpu = (T *) malloc(num_elements * sizeof(T));
    for(int i=0; i<num_elements; i++) eb_gpu[i] = max_pwr_eb;
 
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
    //initialize U and V with padding
    dim3 blockSize(16, 16);
    dim3 gridSize((r2 + (blockSize.x-2) - 1) / (blockSize.x-2), (r1 + (blockSize.y-2) - 1) / (blockSize.y-2));
    printf("gridSize: %d, %d\n", gridSize.x, gridSize.y);
    cudaError_t err;

    T *dU, *dV, *dEb;
    cudaMalloc(&dU, r1 * r2 * sizeof(T));
    cudaMalloc(&dV, r1 * r2 * sizeof(T));
    cudaMalloc(&dEb, r1 * r2 * sizeof(T));
    cudaMemcpy(dU, U, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(dEb, eb_gpu, r1 * r2 * sizeof(T), cudaMemcpyHostToDevice);

    //run kernel
    derive_eb<<<gridSize, blockSize>>>(dU, dV, dEb, r1, r2, max_pwr_eb);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(dU);
        cudaFree(dV);
        cudaFree(dEb);
        return 0;
    }
    
    cudaMemcpy(eb_gpu, dEb, r1 * r2 * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dU);
    cudaFree(dV);
    cudaFree(dEb);
    printf("compute eb_gpu done\n");   

    //for verfiy used
    double diff = 0.0;
    double maxdiff = 0.0;
    int count=0;
    int maxdiff_index = 0;

    for (int i = 0; i < r1*r2; i++){
        diff = fabs(eb_gpu[i] - eb[i]);
        if(diff > maxdiff)
        { 
            maxdiff = diff;
            maxdiff_index = i;    
        }
        if (diff > 1e-7) {
            //printf("error. eb_gpu: %5.2f, eb: %5.2f,%d\n", eb_gpu[i],eb[i],i);
            //break;
            count++;
        }
    }
    printf("maxdiff: %f, maxdiff_index: %d, error count: %d\n", maxdiff, maxdiff_index, count);
    printf("eb_gpu: %f, eb: %f\n", eb_gpu[maxdiff_index], eb[maxdiff_index]);
    
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

   
    free(eb);
    free(eb_gpu); 
    return 0;
}

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
    unsigned char * result = sz_compress_cp_preserve_2d_offline_cpu_gpu(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz_compress_cp_preserve_2d_offline_log(U, V, r1, r2, result_size, false, max_eb);
    // unsigned char * result = sz_compress_cp_preserve_2d_online(U, V, r1, r2, result_size, false, max_eb);
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

    verify(U, dec_U, num_elements);
    verify(V, dec_V, num_elements);

    writefile((string(argv[1]) + ".out").c_str(), dec_U, num_elements);
    writefile((string(argv[2]) + ".out").c_str(), dec_V, num_elements);
    */
    free(result);
    free(U);
    free(V);
    //free(dec_U);
    //free(dec_V);
}