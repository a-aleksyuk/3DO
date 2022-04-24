#include "region.h"

Region::Region() { _type = Everything; }

Region::~Region() {
  if (_type == Rectangle2D) {
    delete[] _rectangle_start;
    delete[] _rectangle_end;
  } else if (_type == Sphere2D) {
    delete[] _sphere_center;
  }
}

void Region::Initialize(json p_input) {
  if (p_input["Type"] == "Rectangle2D") {
    _type = Rectangle2D;
    _dimension = 2;
    _rectangle_start = new double[_dimension];
    _rectangle_end = new double[_dimension];
    for (int i = 0; i < _dimension; i++) {
      _rectangle_start[i] = p_input["Parameters"][0][i];
      _rectangle_end[i] = p_input["Parameters"][1][i];
    }
  } else if (p_input["Type"] == "Sphere2D") {
    _type = Sphere2D;
    _dimension = 2;

    _sphere_center = new double[_dimension];
    for (int i = 0; i < _dimension; i++)
      _sphere_center[i] = p_input["Parameters"][0][i];

    _sphere_radius_2 = p_input["Parameters"][1];
    _sphere_radius_2 *= _sphere_radius_2;
  } else
    _type = Everything;
}

bool Region::IsInRegion(double p_x1, double p_x2) {
  if (_type == Everything)
    return true;
  else if (_type == Rectangle2D) {
    if (_rectangle_start[0] < p_x1 && p_x1 < _rectangle_end[0] &&
        _rectangle_start[1] < p_x2 && p_x2 < _rectangle_end[1])
      return true;
  } else if (_type == Sphere2D) {
    if ((_sphere_center[0] - p_x1) * (_sphere_center[0] - p_x1) +
            (_sphere_center[1] - p_x2) * (_sphere_center[1] - p_x2) <
        _sphere_radius_2)
      return true;
  }

  return false;
}
