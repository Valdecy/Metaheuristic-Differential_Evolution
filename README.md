# Metaheuristic-Differential_Evolution
 Differential Evolution to Minimize Functions with Continuous Variables. The function returns: 1) An array containing the used value(s) for the target function and the output of the target function f(x). For example, if the function f(x1, x2) is used, then the array would be [x1, x2, f(x1, x2)].  


* n = The population size. The Default Value is 3.

* min_values = The minimum value that the variable(s) from a list can have. The default value is -5.

* max_values = The maximum value that the variable(s) from a list can have. The default value is  5.

* iterations = The total number of iterations. The Default Value is 50.

* F = Scaling Factor or Amplification Factor, is a positive real number, typically less than 1.0 thatvcontrols the rate at which the population evolves. The Default Value is 0.9.

* Cr = Crossover Probability [0, 1], in DE the mutation occurs before the crossover. The Default Value is 0.2.

* target_function = Function to be minimized.
