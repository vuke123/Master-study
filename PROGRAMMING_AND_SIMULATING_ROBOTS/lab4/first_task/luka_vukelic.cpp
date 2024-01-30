#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd, Eigen::VectorXd; 

int main() {

    MatrixXd m1(5,5);
    MatrixXd m2(5,5);
    VectorXd v(5);

    int jmbag1[] = {3, 0, 7, 2, 8};
    int jmbag2[] = {0, 0, 3, 6, 5};

    for(int i = 0; i < m1.rows(); i++){
        v[i] = jmbag2[i];
        for(int j = 0; j < m1.cols(); j++){
            m1(i,j) = jmbag1[j];
        }
    }

    m2 = m1 + MatrixXd::Identity(5,5);

    std::cout << m2 * v << std::endl;

    std::cout << v * v.transpose() << std::endl;

    std::cout << m2.determinant() << std::endl;

    std::cout << m2.inverse() << std::endl;

    std::cout << m2.transpose() << std::endl;

    return 0;
}
