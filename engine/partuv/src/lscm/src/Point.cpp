#include "Point.h"
#include <ostream>

using namespace MeshLib;

std::ostream & operator<<( std::ostream & os, const Point & p) {
	os << "Point: " << p(0) << " " << p(1) << " " << p(2) << std::endl;
	return os;
}

Point Point::rotate(double theta, Point vector) {
	Point result;
	double cos_t = cos(theta);
	double sin_t = sin(theta);
    result = vector * (m_v[0] * vector[0] + m_v[1] * vector[1] + m_v[2] * vector[2]) * (1-cos_t);
    result[0] += m_v[0] * cos_t;
    result[1] += m_v[1] * cos_t;
    result[2] += m_v[2] * cos_t;
    result[0] += (m_v[1] * vector[2] - m_v[2] * vector[1]) * sin_t;
    result[1] += (m_v[2] * vector[0] - m_v[0] * vector[2]) * sin_t;
    result[2] += (m_v[0] * vector[1] - m_v[1] * vector[0]) * sin_t;
	return result;
}
