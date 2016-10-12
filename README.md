#Smolyak Sparse Grid Library
This library is an implementation of Smolyak’s Sparse Grid Algorithm for solving integration and interpolation problems
in d-dim spaces with far fewer function evaluations than needed with traditional tensor production
integration/interpolation.

This library currently implements Smolyak's algorithm for two polynomial bases:

    Clenshaw-Curtis – Piecewise Linear Basis Functions (sup. [0,1])

    Chebyshev Polynomials – Cos Basis Functions (sup. [0,1])

    Fejer Polynomials - Coming Soon - (open sup. (-1,1))

I implement two sparse sampling integration rules based on Stroud's Theory (Stroud, 1957) and its extension by
Xiu (Xiu, 2007) that are useful when solving numerical integration problems in n-dimensional Euclidean spaces
with equally and explicitly-weighted points.

    Stroud - Rules for n-dimensional Euclidean spaces of degree two or three (uniform distributions)

    Xiu - Rules for n-dimensional non-symmetric Euclidean spaces of degree two or three (Gaussians)

I also implement several general sampling algorithms for utility :

    Poisson Disk Random - Samples drawn from uniform probability distribution with accept/reject rule for optimal point placement

    Uniform Random - Samples drawn from uniform probability distribution

####Usage####

    /tests/sparse_grid_interp_tests.py - example usage and tests for sparse-grid interpolation
    /tests/grid_random_sampler_tests.py - example usage and tests for sparse-grid, Poisson, and Uniform random samplers

####User defined function####

    fun_nd.py provides example code that evaluates functions in 2D for a given test function. The user must specify a
    new funtion fun_nd.py that follows this same format and evaluates n-dim target functions desired for interpolation.

####Dependencies####
    itertools

####Smolyak’s Sparse Grid Algorithm####

    One approach to solving numerical integration (or polynomial interpolation) is to use 1D Gauss quadrature rules
    (or Gauss-Hermite polynomials) that are applied separately to each dimension, forming a tensor-product rule. However,
    for a 1D rule requiring m function evaluations, the associated tensor-product rule requires m^d evaluations in d dimensions;
    such exponential growth makes this approach computationally intractable for more than a few dimensions. This so-called
    curse of dimensionality is a feature of all tensor-product rules.

    Another method that has received much attention for the evaluation of high-dimensional integrals is Monte-Carlo integration
    (Robert and Casella, 2004). The idea of treating our input parameters as random is logical, and the sampling method can
    be tailored to their known functional forms. The method does not depend formally on the dimension of the random space,
    so it is a seemingly good choice for multivariate integration. However, the method also exhibits slow convergence for
    statistical moments (e.g., sqrt(k) for the mean) and accuracy, which is highly dependent on the exact functional form
    (See Fishman, 1996).

    In this library, I implement Smolyak’s sparse grid method based on appropriate 1D quadrature rules (See Barthelmann
    et al., 2000; Smolyak, 1963). With this method, well-established univariate integration formulas (e.g., Gauss-Quadrature,
    Clenshaw-Curtis, Chebyshev, etc.) are extended to the multivariate case by using a subset of the complete tensor product
    set of abscissae. As a result, we can perform accurate integration (or interpolation) that requires orders of magnitude
    fewer function evaluations than conventional integration on full uniform grids as long as the degree of exactness required
    is less than the dimensionality of the function space. The degree of sparseness that is achievable depends directly on the
    degree to which the quadrature abscissae are “nested” in both space and degree. Unfortunately, many standard quadrature
    rules are poor choices for sparse grid integration (or interpolation), because their rules are weakly nested. For example,
    while Gauss quadrature is generally well-suited for unbounded Gaussian distributions; the formula is not well-nested in
    higher dimensions. There are several other choices of 1D quadrature rules that exhibit highly-nested properties, such
    as the Chebyshev and Clenshaw-Curtis formulae, which for higher dimensions provide rules with orders of magnitude fewer
    points than those for tensor-product rules when the abscissae properties are chosen well (Xiu, 2007).

####References####

    Barthelmann, V., E. Novak, and K. Ritter, 2000, High dimensional polynomial interpolation on sparse grids,
    Adv. in Comput. Math., 12, 273–288.

    Fishman, G., 1996, Monte Carlo: Concepts, Algorithms, and Applications, Springer-Verlag, New York.

    Smolyak, S., 1963, Quadrature and interpolation formulas for tensor products of certain classes of functions,
    Soviet Math. Dokl., 4, 240-243.

    Stroud, A.H., 1957, Remarks on the disposition of points in numerical integration formulas, Math Tables & Other
    Aids in Computing, 257-261.

    Waldvogel, J., 2003, Fast construction of the Fejér and Clenshaw-Curtis quadrature rules, BIT Numerical Mathematics,
    43(1), 1-18.

    Xiu, D., 2007, Efficient collocational approach for parametric uncertainty analysis, 2(2), Commun. Comput. Phys.,
    293-309.

    Xiu, D., 2007, Numerical integration formulas of degree two, Applied Numerical Mathematics (2007),
    doi:10.1016/j.apnum.2007.09.004.


