#ifndef REGION_H
#define REGION_H

#include "json.hpp"

using json = nlohmann::json;

enum RegionType { Everything, Rectangle2D, Sphere2D };

class Region {
 private:
  RegionType _type;
  int _dimension;

  // Rectangle
  double *_rectangle_start;
  double *_rectangle_end;

  // Sphere
  double *_sphere_center;
  double _sphere_radius_2;

 public:
  Region();
  virtual ~Region();

  void Initialize(json p_input_json);
  bool IsInRegion(double p_x1, double p_x2);
};

#endif  // REGION_H
