#ifndef LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_

#include <LightGBM/objective_function.h>

namespace LightGBM {
/*!
* \brief Objective funtion for regression
*/
class RegressionL2loss: public ObjectiveFunction {
public:
  explicit RegressionL2loss(const ObjectiveConfig&) {
  }

  ~RegressionL2loss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const score_t* score, score_t* gradients,
                    score_t* hessians) const override {
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = (score[i] - label_[i]);
        hessians[i] = 1.0;
      }
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = (score[i] - label_[i]) * weights_[i];
        hessians[i] = weights_[i];
      }
    }
  }

  double GetSigmoid() const override {
    // not sigmoid transform, return -1
    return -1.0;
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
};

/*!
* \brief Fair Objective funtion for regression
*/
class RegressionFairloss: public ObjectiveFunction {
public:
  explicit RegressionFairloss(const ObjectiveConfig&) {
  }

  ~RegressionFairloss() {
  }

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    label_ = metadata.label();
    weights_ = metadata.weights();
  }

  void GetGradients(const score_t* score, score_t* gradients,
                    score_t* hessians) const override {
    score_t fair_c_value = 2.0;
    if (weights_ == nullptr) {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = 100 * (score[i] - label_[i]) / ( 1 + std::abs(score[i] - label_[i]) / fair_c_value);
        hessians[i] = 100 * 1.0 / pow( 1 + std::abs(score[i] - label_[i]) / fair_c_value, 2.0) ;
      }
      // for (data_size_t i = 0; i < num_data_; ++i)
      //   Log::Stdout("[%d] %g, %g, %g", i, (score[i] - label_[i]), gradients[i], hessians[i]);
    } else {
      #pragma omp parallel for schedule(static)
      for (data_size_t i = 0; i < num_data_; ++i) {
        gradients[i] = weights_[i] * (score[i] - label_[i]) / ( 1 + std::abs(score[i] - label_[i]) / fair_c_value);
        hessians[i] = weights_[i] / pow( 1 + std::abs(score[i] - label_[i]) / fair_c_value, 2.0) ;
      }
    }
  }

  double GetSigmoid() const override {
    // not sigmoid transform, return -1
    return -1.0;
  }

private:
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const float* label_;
  /*! \brief Pointer of weights */
  const float* weights_;
};
}  // namespace LightGBM
#endif   // LightGBM_OBJECTIVE_REGRESSION_OBJECTIVE_HPP_
