# Experiments

This section contains the experiments done to compare the different gradient propagation strategies implemented in `vqrotate`.

The experiments use the Speech Commands dataset and compare:

- Straight Through Estimator (STE)
- Rotation Trick
- Adaptive Scaling *(experimental)*

The purpose of these experiments is to observe how the different strategies affect reconstruction loss, codebook usage, and codebook perplexity.

## 1. Experimental Setup

### Dataset

The experiments were performed using the **Speech Commands** dataset.

### Strategies

The following strategies were evaluated:

| Strategy | Description |
|----------|-------------|
| STE | Straight Through Estimator |
| Rotation | Rotation Trick |
| Adaptive | Adaptive Scaling *(experimental)* |

### Metrics

The following metrics were recorded during training:

- **Reconstruction MSE**
- **Codebook Usage**
- **Codebook Perplexity**
- **Assignment Flip Rate**

---
# Experiment 1

## Overview

The first experiment was performed on the Speech Commands dataset to compare
the behaviour of the three gradient propagation strategies:

- STE
- Rotation Trick
- Adaptive Scaling

All three strategies were trained under the same general setup so that their
training behaviour could be compared.

The main focus of this experiment was **codebook utilization** and how the
different gradient propagation strategies affected the VQ layer during
training.

Note that it was a small experiment with 10 epoches.

## Results

The final codebook usage was:

| Strategy | Final Codebook Usage |
|----------|---------------------:|
| STE | 19.99% |
| Rotation | **32.88%** |
| Adaptive Scaling | 22.98% |

Rotation achieved the highest final codebook usage in this experiment.
## Observations
The codebook usage changed significantly during the first few epochs.

STE started around 40% usage before settling around the 20% range.

Rotation started with substantially higher usage and remained higher than the other strategies for much of the training process. It ended with 32.88% of the codebook being used.

Adaptive Scaling initially behaved similarly to STE, but finished at 22.98%.

The main observation from this experiment is that the Rotation Trick produced a noticeably different codebook utilization pattern from STE.

This suggests that changing the way gradients are propagated through the quantization operation can affect how the model makes use of the codebook.

But higher codebook usage does not automatically mean better model
performance. It is only one measurement of the behaviour of the VQ layer.

## Conclusion

The main observation from Experiment 1 is that the Rotation Trick produced
substantially higher final codebook utilization than STE and Adaptive Scaling.

At the same time, Adaptive Scaling showed interesting behaviour in
reconstruction loss and perplexity.

These results motivated running another experiment rather than treating this
single run as a definitive comparison.

# Experiment 2

## Overview

The second experiment was another benchmark on the Speech Commands dataset.

The same three gradient propagation strategies were compared:

- STE
- Rotation Trick
- Adaptive Scaling

The purpose of the second experiment was to see whether the behaviour
observed in the first experiment would also appear in another training run plus it measures more metrics in particular.

## Results

The final results were:

| Strategy | Codebook Usage | MSE | Perplexity |
|----------|---------------:|----:|-----------:|
| STE | 21.42% | 13.8839 | 9.87 |
| Rotation | 21.09% | 15.2902 | 15.57 |
| Adaptive | 20.51% | **13.5147** | **36.50** |


Unlike Experiment 1, Rotation did not produce substantially higher final
codebook usage in this run.

The final usage values were very close:

- STE: 21.42%
- Rotation: 21.09%
- Adaptive: 20.51%

All strategies produces similarish results in utilization
This is an important difference from Experiment 1, where Rotation reached
32.88% usage compared with 19.99% for STE.

## Reconstruction Loss


Adaptive Scaling produced the lowest final reconstruction loss:

$$
\text{Adaptive} = 13.5147
$$

compared with:

$$
\text{STE} = 13.8839
$$

and:

$$
\text{Rotation} = 15.2902
$$

The Adaptive Scaling run also showed a relatively consistent decrease in
reconstruction loss throughout training.

This is an interesting result because Adaptive Scaling was introduced as an
experimental extension of the Rotation Trick rather than as a separately
validated method.

This suggests that Adaptive Scaling may be trading codebook exploration for reconstruction precision, compressing the audio into a smaller set of highly effective codes.

## Codebook Perplexity


The difference in perplexity between the strategies was not some thing I was expecting.

At the end of training:

| Strategy | Perplexity |
|----------|-----------:|
| STE | 9.87 |
| Rotation | 15.57 |
| Adaptive | **36.50** |

Adaptive Scaling showed a strong increase in perplexity throughout the
experiment.

This indicates that Adaptive Scaling produced substantially different
codebook usage behaviour from both STE and Rotation in this run.

## Codebook Usage


The codebook usage curves show that all three strategies experienced a large
change in utilization during the early epochs.

After that initial change, the strategies behaved differently.

Rotation had relatively high usage during the first epoch, but this advantage
did not remain throughout training.

Adaptive Scaling finished at 20.51%, while STE finished at 21.42%.

The complete experiment log records the final usage values as 21.42% for STE,
21.09% for Rotation, and 20.51% for Adaptive Scaling. 

## Assignment Flip Rate


The measured flip rate became very high during training, particularly for
Adaptive Scaling.

However, this metric should **not** be interpreted as evidence that Adaptive
Scaling has unstable assignments.

The current implementation measures changes between consecutive training
batches. Since the Speech Commands dataset is shuffled, consecutive batches
can contain completely different audio samples.

Hence, a high flip rate can simply mean that different samples are being
assigned to different codes.

you can check out the benchmark code on my github also

For example:

```text
Batch 0 → dog bark
Batch 1 → baby crying
```
Comparing the assignments of these two different batches does not tell us
whether the assignment of a particular sample is stable.

A proper assignment-stability experiment would evaluate a fixed validation
set after every epoch and compare the assignments of the exact same samples.

For this reason, the flip-rate values in this experiment are reported as
observations but are not used to claim that one method is more stable than
another.

## Conclusion

Experiment 2 produced a different result from Experiment 1.

Rotation did not reproduce the large codebook-usage advantage observed in
Experiment 1.

Instead, Adaptive Scaling achieved the lowest reconstruction MSE and the
highest perplexity in this run.

Hence, these experiments should be treated as behavioural experiments
rather than a definitive benchmark proving that one strategy is universally
better than another.

# Limitations

These experiments are limited to a single audio dataset, a fixed codebook size (512), and 10 training epochs. The results should be interpreted as preliminary observations rather than a definitive comparison. Future work should include larger codebooks, more epochs, and multiple random seeds to produce statistically robust averages.

MORE EXPERIMENTS ARE IN THE WAY MAYBE ;)
