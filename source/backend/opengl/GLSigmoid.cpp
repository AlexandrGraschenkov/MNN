//
// Created by Alexander Graschenkov on 07.12.2020.
//

#include "backend/opengl/GLSigmoid.hpp"
#include <sstream>
#include "AllShader.hpp"
#include "backend/opengl/GLBackend.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"

namespace MNN {
namespace OpenGL {
GLSigmoid::GLSigmoid(const std::vector<Tensor *> &inputs, const Op *op, Backend *bn) : Execution(bn) {
  mType = op->type();
}

GLSigmoid::~GLSigmoid() {

}

ErrorCode GLSigmoid::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
  std::vector<std::string> prefix;
  setLocalSize(prefix, mLocalSize, 8, 8, 1);
  if (OpType_Sigmoid == mType) {
    mProgram = ((GLBackend *)backend())->getProgram("sigmoid", glsl_sigmoid_glsl, prefix);
  }else{
    MNN_PRINT("not support !!!");
    return NOT_SUPPORT;
  }
  return NO_ERROR;
}

ErrorCode GLSigmoid::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
  auto input = inputs[0];
  auto output = outputs[0];
  int iw = input->width();
  int ih = input->height();
  int ic_4 = UP_DIV(input->channel(), 4);
  int ib = input->batch();

  mProgram->useProgram();
  glBindImageTexture(0, output->deviceId(), 0, GL_TRUE, 0, GL_WRITE_ONLY, ((GLBackend *)backend())->getTextrueFormat());
  {
    int texId = 0;
    glActiveTexture(GL_TEXTURE0 + texId);
    glUniform1i(1, texId);
    glBindTexture(GL_TEXTURE_3D, input->deviceId());
    OPENGL_CHECK_ERROR;
  }
  glUniform4i(2, iw, ih, ic_4, ib);
  OPENGL_CHECK_ERROR;
  ((GLBackend *)backend())->compute(UP_DIV(iw, mLocalSize[0]), UP_DIV(ih, mLocalSize[1]), UP_DIV(ic_4, mLocalSize[2]));

  return NO_ERROR;
}
GLCreatorRegister<TypedCreator<GLSigmoid>> __sigmoid_op(OpType_Sigmoid);
} // namespace OpenGL
} // namespace MNN
