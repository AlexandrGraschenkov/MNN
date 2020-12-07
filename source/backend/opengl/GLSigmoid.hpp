//
// Created by Alexander Graschenkov on 07.12.2020.
//


#ifndef GLSigmoid_H
#define GLSigmoid_H
#include "core/Execution.hpp"
#include "backend/opengl/GLProgram.hpp"
#include "backend/opengl/GLTexture.hpp"
#include "backend/opengl/GLSSBOBuffer.hpp"
#include "MNN_generated.h"
namespace MNN {
namespace OpenGL {
class GLSigmoid : public MNN::Execution {
 public:
  GLSigmoid(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn);
  virtual ~GLSigmoid();
  ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

 private:
  std::shared_ptr<GLProgram> mProgram;
  std::shared_ptr<GLSSBOBuffer> mSlopeBuffer;
  int mLocalSize[3];
  int mType;
  const Op* mOp;
};
} // namespace OpenGL
} // namespace MNN

#endif // GLRelu_H
