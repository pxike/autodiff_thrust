
#include <queue>
#include "cuda_runtime.h"
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

struct ConvolutionRev {
    double* result_values;
    double* A_grad;
    double* kernel_values;
    double* kernel_grad;
    size_t num_rows_A;
    size_t num_cols_A;
    size_t num_rows_result;
    size_t num_cols_result;
    size_t kernelHeight;
    size_t kernelWidth;

    ConvolutionRev(double* _result_values, double* _A_grad, double* _kernel_values, double* _kernel_grad,
        size_t _num_rows_A, size_t _num_cols_A, size_t _num_rows_result,
        size_t _num_cols_result, size_t _kernelHeight, size_t _kernelWidth)
        : result_values(_result_values), A_grad(_A_grad), kernel_values(_kernel_values),
        kernel_grad(_kernel_grad), num_rows_A(_num_rows_A), num_cols_A(_num_cols_A),
        num_rows_result(_num_rows_result), num_cols_result(_num_cols_result),
        kernelHeight(_kernelHeight), kernelWidth(_kernelWidth) {}

    __device__
        void operator()(size_t idx) const {
        size_t i = idx % num_cols_result; // Get column index
        size_t j = idx / num_cols_result; // Get row index

        // Get the value from the result tensor
        double result_value = result_values[j * num_cols_result + i];

        for (size_t k = 0; k < kernelHeight; ++k) {
            for (size_t l = 0; l < kernelWidth; ++l) {
                size_t row_A = j + k;
                size_t col_A = i + l;

                if (row_A < num_rows_A && col_A < num_cols_A) {
                    // Multiply the kernel value with the result value
                    double kernel_value = kernel_values[k * kernelWidth + l];
                    A_grad[row_A * num_cols_A + col_A] += kernel_value * result_value;

                    // Update the kernel gradient
                    kernel_grad[k * kernelWidth + l] += A_grad[row_A * num_cols_A + col_A] * result_value / (num_cols_A* num_rows_A);
                }
            }
        }
    }
};


struct MulTY
{
    const double* A,* B;
    size_t num_cols_A;
    size_t num_cols_B;

    MulTY(const double* A,
        const double* B,
        size_t num_cols_A, size_t num_cols_B)
        : A(A), B(B), num_cols_A(num_cols_A), num_cols_B(num_cols_B) {}

    __device__
        double operator()(size_t idx) const {
        double sum = 0;
        size_t row = idx / num_cols_B;
        size_t col = idx % num_cols_B;
        for (size_t i = 0; i < num_cols_A; ++i) {
            sum += A[num_cols_A * row + i] * B[i * num_cols_B + col];
        }
        return sum;
    }
};
struct REVmulB
{
    const double * B;
    size_t num_cols_A;
    size_t num_cols_B;

    REVmulB(const double* B,
        size_t num_cols_A, size_t num_cols_B)
        : B(B), num_cols_A(num_cols_A), num_cols_B(num_cols_B) {}

    __device__
        double operator()(size_t idx) const {
        double sum = 0;
        size_t col = idx % num_cols_B;
        for (size_t i = 0; i < num_cols_A; ++i) {
            sum += B[i * num_cols_B + col];
        }
        return sum;
    }
};
struct REVmulA
{
    const double* A;
    size_t num_cols_A;
    size_t num_cols_B;

    REVmulA(const double* A,
        size_t num_cols_A, size_t num_cols_B)
        : A(A), num_cols_A(num_cols_A), num_cols_B(num_cols_B) {}

    __device__
        double operator()(size_t idx) const {
        double sum = 0;
        size_t row = idx / num_cols_B;
        for (size_t i = 0; i < num_cols_A; ++i) {
            sum += A[num_cols_A * row + i];
        }
        return sum;
    }
};
struct REVAddition {
    REVAddition(){}
    __host__ __device__
        void operator()(thrust::tuple<double&, double&, double> t) const {
        thrust::get<0>(t) += thrust::get<2>(t); // Add the corresponding elements to vector1
        thrust::get<1>(t) += thrust::get<2>(t); // Add the corresponding elements to vector2
    }
};

struct MultiplyValue {
    double multiplier;

    MultiplyValue(double _multiplier) : multiplier(_multiplier) {}

    __host__ __device__
        double operator()(const double& element) const {
        return element * multiplier;
    }
};

void accumulate_elements(thrust::device_vector<double>& destination, int index, const thrust::device_vector<double>& source) {
    thrust::transform(destination.begin() + index * source.size(),
        destination.begin() + (index + 1) * source.size(),
        source.begin(),
        destination.begin() + index * source.size(),
        thrust::plus<double>());
}

thrust::device_vector<double> extract_column(const thrust::device_vector<double>& matrix, int column_index, int num_rows) {
    thrust::device_vector<double> column(num_rows);
    thrust::copy(matrix.begin() + column_index * num_rows,
        matrix.begin() + (column_index + 1) * num_rows,
        column.begin());
    return column;
}

thrust::device_vector<double> extract_row(const thrust::device_vector<double>& matrix, int row_index, int num_cols) {
    thrust::device_vector<double> row(num_cols);
    thrust::copy(matrix.begin() + row_index * num_cols,
        matrix.begin() + (row_index + 1) * num_cols,
        row.begin());
    return row;
}


struct Conv {
    const double* input;
    const double* kernel;
    size_t inputWidth;
    size_t inputHeight;
    size_t kernelWidth;
    size_t kernelHeight;

    Conv(const double* input,
        const double* kernel,
        size_t inputWidth, size_t inputHeight, size_t kernelWidth, size_t kernelHeight)
        : input(input), kernel(kernel), inputWidth(inputWidth), inputHeight(inputHeight),
        kernelWidth(kernelWidth), kernelHeight(kernelHeight) {}

    __device__
        double operator()(size_t idx) const {
        size_t col = idx % (inputWidth - kernelWidth + 1);
        size_t row = idx / (inputWidth - kernelWidth + 1);

        if (row >= inputHeight || col >= inputWidth) {
            return 0;
        }
        double sum = 0;
        for (size_t kRow = 0; kRow < kernelHeight; ++kRow) {
            for (size_t kCol = 0; kCol < kernelWidth; ++kCol) {
                size_t inputRow = row + kRow;
                size_t inputCol = col + kCol;
                size_t inputIdx = inputRow * inputWidth + inputCol;
                size_t kernelIdx = kRow * kernelWidth + kCol;
                if (inputRow < inputHeight && inputCol < inputWidth) {
                    sum += input[inputIdx] * kernel[kernelIdx];
                }
            }
        }
        return sum;
    }
};

class Tensor {

public:
    enum Reversal
    {
	    none,add,mul,relu,conv,matmul
    };
    thrust::device_vector<double> value;
    thrust::device_vector<double> grad;
	std::vector<size_t> shape;
    std::vector<Tensor*> children;
    Reversal operation;

    Tensor(const thrust::host_vector<double>& host_values,
        const std::vector<size_t>& input_shape,
        const std::vector<Tensor*>& children = std::vector<Tensor*>(2),
        Reversal op = Reversal::none) : shape(input_shape), operation(op) {
        value = thrust::device_vector<double>(host_values);
        grad = thrust::device_vector<double>(host_values.size(), 0.0);

    }

    // Constructor using thrust::host_vector for all parameters
    Tensor(const thrust::host_vector<double>& val = {},
        Reversal op = Reversal::none,
        const std::vector<Tensor*>& children = std::vector<Tensor*>(2)
    )
        : value(val.begin(), val.end()),
        grad(thrust::device_vector<double>(val.size(), 0.0)),
        operation(op),
        shape(std::vector<size_t>()) {shape.push_back(val.size());}

    Tensor operator+( Tensor& other)  {
        // Check if shapes are compatible for addition
        if (shape != other.shape) {
            throw std::runtime_error("Shapes are not compatible for addition.");
        }
        Tensor result;
        result.children = { this, &other };
		
        result.value.resize(value.size());
        result.grad.resize(grad.size());
        result.operation = Reversal::add;
        result.shape = shape;
        // Perform element-wise addition
        thrust::transform(value.begin(), value.end(), other.value.begin(), result.value.begin(), thrust::plus<double>());
        return result;
    }

    Tensor operator*( Tensor& other)  {
        Tensor result;
        result.children = { this, &other };
        if (shape.size() == 1 && other.shape.size() == 1 && shape[0] == other.shape[0]) {

            result.shape = shape;
            result.value.resize(shape[0]);
            result.grad.resize(shape[0]);

            result.operation = Reversal::mul;

            thrust::transform(
				value.begin(), value.end(),
                other.value.begin(),
                result.value.begin(),
                thrust::multiplies<double>()
            );
            result.children = { this, &other };
            return result;
        }

        if (shape.size() < 3 && other.shape.size() < 3 && shape[1] == other.shape[0]) {
            size_t numRows = shape[0];
            size_t numCols = other.shape.size() == 1 ? 1 : other.shape[1];
            size_t innerDim = shape[1];

            if (other.shape.size() != 1)
            {
                result.shape.resize(2);
                result.shape = { shape[0],other.shape[1] };

            }
        	else
            {
                result.shape = { shape[0] };
            }
            result.value.resize(numRows * numCols);
            result.grad.resize(numRows * numCols);

            // Perform matrix multiplication using the MulTY functor.
            MulTY multKernel(
                (value.data().get()),
                (other.value.data().get()),
                innerDim,
                numCols
            );

            thrust::device_vector<double> resultData(numRows * numCols);
            thrust::device_vector<double>::iterator resultDataEnd = thrust::transform(
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(numRows * numCols),
                resultData.begin(),
                multKernel
            );
            result.value = resultData;
            result.operation = Reversal::matmul;
            return result;
        }


        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");

    }

    Tensor operator/( Tensor& kernel) 
    {

        if (shape.size() != 2 || kernel.shape.size() != 2) {
            throw std::invalid_argument("Both tensor and kernel must be 2D.");
        }

        size_t inputHeight = shape[0];
        size_t inputWidth = shape[1];
        size_t kernelHeight = kernel.shape[0];
        size_t kernelWidth = kernel.shape[1];
        size_t outputHeight = inputHeight - kernelHeight + 1;
        size_t outputWidth = inputWidth - kernelWidth + 1;

        thrust::device_vector<double> resultData(outputHeight * outputWidth);
        double* resultDataRaw = raw_pointer_cast(resultData.data());

        Conv convKernel(
            thrust::raw_pointer_cast(value.data()), thrust::raw_pointer_cast(kernel.value.data()),
            inputWidth, inputHeight, kernelWidth, kernelHeight);

        thrust::transform(thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(outputHeight * outputWidth),
            resultData.begin(),
            convKernel);

        Tensor result;
        result.grad = thrust::device_vector<double>(resultData.size(), 0.0);
        result.children = { this, &kernel };
        result.value = resultData;
        result.shape = { outputHeight, outputWidth };
        result.operation = Reversal::conv;

        return result;

    }
    void backward()
    {
        thrust::fill(grad.begin(), grad.end(), 1.0);
        std::queue<Tensor*> q;
        q.push(this);
        while (!q.empty()) {
            Tensor* current = q.front();
            current->print2();
            q.pop();

            switch (current->operation) {
            case Reversal::add: {
                thrust::for_each(
                    thrust::make_zip_iterator(thrust::make_tuple(current->children[1]->grad.begin(), current->children[0]->grad.begin(), current->grad.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(current->children[1]->grad.end(), current->children[0]->grad.end(), current->grad.end())),
                    REVAddition()
                );
                break;
            }
            case Reversal::mul: {
                thrust::device_vector<double> first(current->value.size());
                thrust::device_vector<double> second(current->value.size());

                thrust::transform(
                    current->grad.begin(), current->grad.end(),
                    current->children[1]->value.begin(),
                    first.begin(),
                    thrust::multiplies<double>()
                );
                thrust::transform(
                    current->grad.begin(), current->grad.end(),
                    current->children[0]->value.begin(),
                    second.begin(),
                    thrust::multiplies<double>()
                );
                thrust::transform(
                    current->children[0]->grad.begin(), current->children[0]->grad.end(),
                    first.begin(),
                    current->children[0]->grad.begin(),
                    thrust::plus<double>()
                );
                thrust::transform(
                    current->children[1]->grad.begin(), current->children[1]->grad.end(),
                    second.begin(),
                    current->children[1]->grad.begin(),
                    thrust::plus<double>()
                );
                break;
            }
            case Reversal::relu: {
                break;
            }
            case Reversal::conv: {
                Tensor* A = current->children[0];
                Tensor* kernel = current->children[1];
                size_t num_rows_A = A->shape[0];
                size_t num_cols_A = A->shape[1];
                size_t num_rows_result = current->shape[0];
                size_t num_cols_result = current->shape[1];
                size_t kernelHeight = kernel->shape[0];
                size_t kernelWidth = kernel->shape[1];

                ConvolutionRev updateFunc(current->value.data().get(), A->grad.data().get(),
                    kernel->value.data().get(), kernel->grad.data().get(),
                    num_rows_A, num_cols_A, num_rows_result, num_cols_result,
                    kernelHeight, kernelWidth);

                // Calculate the total number of elements in the result matrix
                size_t total_elements = num_rows_result * num_cols_result;

                thrust::for_each_n(
                    thrust::counting_iterator<size_t>(0),
                    total_elements,
                    updateFunc
                );
                break;
            }
            case Reversal::matmul: {
                size_t num_rows_A = current->children[0]->shape[0];
                size_t num_cols_B = current->children[1]->shape.size() == 1 ? 1 : current->children[1]->shape[1] ;
                size_t num_rows_C = current->shape[0];
                size_t num_cols_C = current->shape.size() == 1 ? 1 : current->shape[1];

                thrust::device_vector<double> C_prime = current->grad; // Assuming C' is stored in current->grad

                for (size_t i = 0; i < num_rows_C; ++i) {
                    for (size_t j = 0; j < num_cols_C; ++j) {
                        thrust::device_vector<double> column_B_j = extract_column(current->children[1]->value, j, num_cols_B);
                        thrust::device_vector<double> row_A_i = extract_row(current->children[0]->value, i, num_cols_B);

                        double grad_ij = C_prime[i * num_cols_C + j]; // Gradient value at (i, j) of the result

                        thrust::transform(column_B_j.begin(), column_B_j.end(), column_B_j.begin(),
                            MultiplyValue(grad_ij)); // Multiply the gradient value by elements of column B_j

                        thrust::transform(row_A_i.begin(), row_A_i.end(), row_A_i.begin(),
                            MultiplyValue(grad_ij)); // Multiply the gradient value by elements of row A_i

                        accumulate_elements(current->children[0]->grad, i, column_B_j); // Accumulate gradient for A
                        accumulate_elements(current->children[1]->grad, j, row_A_i); // Accumulate gradient for B
                    }
                }
                break;
            }
            case Reversal::none: {
                break;
            }
            }
            if (!current->children.empty()) {
                for (Tensor* child : current->children) {
                    q.push(child);
                }
            }
        }
    }
    void print2() const {
        std::cout << "Value: ";
        for (size_t i = 0; i < value.size(); ++i) {
            std::cout << value[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Gradient: ";
        for (size_t i = 0; i < grad.size(); ++i) {
            std::cout << grad[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Shape: ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Operation: " << operation << std::endl;
    }
    void print() const {

        thrust::host_vector<double> fil = this->value;
        thrust::host_vector<double> filg = this->grad;

        std::vector<size_t> leftProduct(shape.begin(), shape.end());

        for (size_t i = 0; i < leftProduct.size() - 1; ++i) {
            leftProduct[i + 1] = leftProduct[i] * leftProduct[i + 1];
        }
        size_t i = 0;
        for (size_t i = 0; i < fil.size(); ++i) {
            bool xx = true;
            printf(" (%0.0f,%0.0f)", fil[i] , filg[i]);
            i += 1;
            for (size_t l : leftProduct) {
                if (i % l == 0)
                {
                    printf("\n");
                    xx = false;
                }
            }
            if (xx) printf(" ");
        }
    }
};

void Tensor_operations_tests() {
    // Initialize tensors b and c as 3x3 matrices
    thrust::host_vector<double> b_data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    thrust::host_vector<double> c_data = { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0 };

    Tensor b(b_data, { 3, 3 });
    Tensor c(c_data, { 3, 3 });

    // Perform operations
    Tensor a = b * c;

    // Initialize kernel tensor k as a 2x2 matrix
    thrust::host_vector<double> k_data = { 1.0, 0.0, 0.0, 1.0 };
    Tensor k(k_data, { 2, 2 });
    Tensor d = a /k ;

    thrust::host_vector<double> l(9, 2.0); // Filled with 2.0 for example
    Tensor newMatrix(l, { 3, 3 });

    Tensor f = b * newMatrix; // f is the multiplication of matrices b and newMatrix

    // Backward pass on the last operation in the chain (f = b * newMatrix)
    f.backward();


}


int main() {
    Tensor_operations_tests();
}
