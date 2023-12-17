## TODO
XLA
* Better error handling (catch cpp errors?).
* More operators.
    * Reductions for logsumexp.

Tracing/autodiff
* Operators
    * Dot-general (autodiff?)
    * Generalize transpose to permutations
    * Reshape
    * Reductions (sum, max)
    * Stop gradient
* Testing - random tests for gradients.

Tensors:
* Getter.
* Remove primitive type where not needed.

# Done
XLA
* infra (client, xla builder, ops. literals, buffers). For now only CPU.
* matrix multiplication.
* Ops: 
    * reshape.
    * transpose.

Tracing/autodiff
* Eval for scalars
    * Mul, add
* Diff for scapars
    * Mul, add