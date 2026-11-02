//! Kernel parameters, compiled for both the GPU and the host. Holds `extern`
//! structs only and imports nothing, so that the two sides cannot disagree.

/// Parameters of `gemm`: c = a * b, for row-major f32 matrices of rows x inner,
/// inner x cols and rows x cols.
pub const Gemm = extern struct {
    a: u64,
    b: u64,
    c: u64,
    rows: u32,
    cols: u32,
    inner: u32,
};

/// One row of the metrics that `softmax_grad` writes.
pub const RowMetrics = extern struct {
    loss: f32,
    correct: u32,
};

/// output = relu(input * weight + bias), or without relu when `relu` is 0.
pub const DenseForward = extern struct {
    input: u64,
    weight: u64,
    bias: u64,
    output: u64,
    rows: u32,
    cols: u32,
    inner: u32,
    relu: u32,
};

/// Writes softmax - one_hot(label) to `grad`, and per-row metrics.
pub const SoftmaxGrad = extern struct {
    logits: u64,
    labels: u64,
    grad: u64,
    metrics: u64,
    rows: u32,
    classes: u32,
};

/// The gradients of one dense layer, given the gradient at its output.
pub const DenseBackward = extern struct {
    input: u64,
    output: u64,
    grad_output: u64,
    weight: u64,
    /// One of: gradient of weight, gradient of bias, or gradient at the input.
    result: u64,
    rows: u32,
    cols: u32,
    inner: u32,
    relu: u32,
};

/// velocity = momentum * velocity + grad, then param -= rate * velocity.
pub const SgdStep = extern struct {
    param: u64,
    grad: u64,
    velocity: u64,
    count: u32,
    rate: f32,
    momentum: f32,
};

/// Parameters of `fill3d`. `out` is a GPU virtual address of width * height * depth
/// f32 elements.
pub const Fill3d = extern struct {
    out: u64,
    width: u32,
    height: u32,
    scale: f32,
    offset: f32,
};
