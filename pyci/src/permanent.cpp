/* This file is part of PyCI.
 *
 * PyCI is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * PyCI is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PyCI. If not, see <http://www.gnu.org/licenses/>. */

#include <pyci.h>

namespace pyci {

double permanent(const double *matrix, const long n) {
    // Permanent of zero-by-zero matrix is 1
    if (n == 0) {
        return 1;
    }
    // Iterate over c = pow(2, n) (equal to (1 << n)) submatrices
    double sum = 0, rowsum, rowsumprod;
    unsigned long c = 1UL << n, k;
    long i, j;
    for (k = 1; k != c; ++k) {
        // Loop over columns of submatrix; compute product of row sums
        rowsumprod = 1;
        for (i = 0; i != n; ++i) {
            // Loop over rows of submatrix; compute row sum
            rowsum = 0;
            for (j = 0; j != n; ++j) {
                // Add element to row sum if the row index is in the characteristic
                // vector of the submatrix, which is the binary vector given by k.
                if (k & (1UL << j)) {
                    rowsum += matrix[n * i + j];
                }
            }
            // Update product of row sums
            rowsumprod *= rowsum;
        }
        // Add term multiplied by the parity of the characteristic vector
        sum += rowsumprod * (1 - ((__builtin_popcountl(k) & 1) << 1));
    }
    // Return answer with the correct sign (times -1 for odd n)
    return (n % 2) ? -sum : sum;
}

double permanent(const double *matrix, const long *cols, const long n, const long rstride) {
    // Permanent of zero-by-zero matrix is 1
    if (n == 0) {
        return 1;
    }
    // Iterate over c = pow(2, n) (equal to (1 << n)) submatrices
    double sum = 0, rowsum, rowsumprod;
    unsigned long c = 1UL << n, k;
    long i, j;
    for (k = 1; k != c; ++k) {
        // Loop over columns of submatrix; compute product of row sums
        rowsumprod = 1;
        for (i = 0; i != n; ++i) {
            // Loop over rows of submatrix; compute row sum
            rowsum = 0;
            for (j = 0; j != n; ++j) {
                // Add element to row sum if the row index is in the characteristic
                // vector of the submatrix, which is the binary vector given by k.
                if (k & (1UL << j)) {
                    rowsum += matrix[rstride * i + cols[j]];
                }
            }
            // Update product of row sums
            rowsumprod *= rowsum;
        }
        // Add term multiplied by the parity of the characteristic vector
        sum += rowsumprod * (1 - ((__builtin_popcountl(k) & 1) << 1));
    }
    // Return answer with the correct sign (times -1 for odd n)
    return (n % 2) ? -sum : sum;
}

double permanent(const double *matrix, const long *rows, const long *cols, const long n,
                 const long rstride) {
    // Permanent of zero-by-zero matrix is 1
    if (n == 0) {
        return 1;
    }
    // Iterate over c = pow(2, n) (equal to (1 << n)) submatrices
    double sum = 0, rowsum, rowsumprod;
    unsigned long c = 1UL << n, k;
    long i, j;
    for (k = 1; k != c; ++k) {
        // Loop over columns of submatrix; compute product of row sums
        rowsumprod = 1;
        for (i = 0; i != n; ++i) {
            // Loop over rows of submatrix; compute row sum
            rowsum = 0;
            for (j = 0; j != n; ++j) {
                // Add element to row sum if the row index is in the characteristic
                // vector of the submatrix, which is the binary vector given by k.
                if (k & (1UL << j)) {
                    rowsum += matrix[rstride * rows[i] + cols[j]];
                }
            }
            // Update product of row sums
            rowsumprod *= rowsum;
        }
        // Add term multiplied by the parity of the characteristic vector
        sum += rowsumprod * (1 - ((__builtin_popcountl(k) & 1) << 1));
    }
    // Return answer with the correct sign (times -1 for odd n)
    return (n % 2) ? -sum : sum;
}

} // namespace pyci
