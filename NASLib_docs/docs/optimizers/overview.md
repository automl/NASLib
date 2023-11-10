# Optimizers Documentation Overview

This documentation provides information on a range of optimizers available in NASLib. Explore these optimizers for both discrete and one-shot architecture search methods, each tailored to specific tasks within the NAS field.

## Discrete Optimizers

### 1. [Bananas](discrete/bananas.md)
Bananas is a powerful discrete optimizer that can efficiently search for neural architectures in a discrete search space. It employs a combination of Bayesian optimization and neural networks to find optimal architectures.

### 2. [Base Predictor](discrete/bp.md)
The Base Predictor optimizer is a foundational component for discrete NAS. It provides a starting point for architecture search by predicting the performance of candidate architectures based on historical data.

### 3. [Local Search](discrete/ls.md)
Local Search is an optimizer that focuses on refining neural architectures through local exploration. It is a valuable tool for fine-tuning architectures to achieve better performance.

### 4. [NPenas](discrete/npenas.md)
NPenas is an optimizer that leverages penalization-based techniques to search for neural architectures. It helps prevent overfitting while finding architectures that perform well on your specific task.

### 5. [Regularized Evolution](discrete/re.md)
Regularized Evolution is a discrete optimizer that uses evolutionary algorithms with regularization to discover high-performing neural architectures. It balances exploration and exploitation to find optimal solutions.

### 6. [Random Search](discrete/rs.md)
Random Search is a simple yet effective optimizer that explores neural architecture search spaces by randomly sampling and evaluating architectures. It serves as a baseline for NAS experiments.

## One-Shot Optimizers

### 7. [DARTS](oneshot/darts.md)
DARTS (Differentiable Architecture Search) is a one-shot optimizer that uses gradient-based methods to search for optimal neural architectures. It allows the continuous relaxation of the architecture search space for efficiency.

### 8. [DrNAS](oneshot/drnas.md)
DrNAS is an optimizer that incorporates regularization techniques into one-shot NAS. It focuses on discovering robust neural architectures by adding regularization constraints during the search process.

### 9. [GDAS](oneshot/gdas.md)
GDAS (Gradient-Driven Architecture Search) is a one-shot optimizer that emphasizes the use of gradients to guide the search for neural architectures. It efficiently explores the search space while maintaining differentiability.

### 10. [RSWS](oneshot/rsws.md)
 RSWS (Random Search with Weight Sharing) is a one-shot optimizer that combines random search with weight sharing to discover neural architectures efficiently. It leverages shared weights to speed up the search process.
