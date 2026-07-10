#ifndef _MESHLIB_POINT_H_
#define _MESHLIB_POINT_H_

#include <cassert>
#include <cmath>

namespace MeshLib  {
	class Point {
	public:
		Point(double x, double y, double z) { m_v[0] = x; m_v[1] = y; m_v[2] = z; };
		Point() { m_v[0] = m_v[1] = m_v[2] = 0; };
		~Point() {};

		double &operator[](int i)       { assert(0 <= i && i < 3); return m_v[i]; };
		double  operator()(int i) const { assert(0 <= i && i < 3); return m_v[i]; };
		double  operator[](int i) const { assert(0 <= i && i < 3); return m_v[i]; };

		// l2-norm
		double norm() const { return sqrt(m_v[0] * m_v[0] + m_v[1] * m_v[1] + m_v[2] * m_v[2]); };
		// lp-norm
		double norm(int i) const { return pow(pow(m_v[0], i) + pow(m_v[1], i) + pow(m_v[2], i), 1 / double(i)); };

		Point &operator += (const Point &p) { m_v[0] += p[0]; m_v[1] += p[1]; m_v[2] += p[2]; return *this; };
		Point &operator -= (const Point &p) { m_v[0] -= p[0]; m_v[1] -= p[1]; m_v[2] -= p[2]; return *this; };
		Point &operator *= (const double s) { m_v[0] *= s;    m_v[1] *= s;    m_v[2] *= s;    return *this; };
		Point &operator /= (const double s) { m_v[0] /= s;    m_v[1] /= s;    m_v[2] /= s;    return *this; };

		Point operator+(const Point &p) const { Point np(m_v[0] + p[0], m_v[1] + p[1], m_v[2] + p[2]); return np; };
		Point operator-(const Point &p) const { Point np(m_v[0] - p[0], m_v[1] - p[1], m_v[2] - p[2]); return np; };
		// Point operator*(const Point &p) const { Point np(m_v[0] * p[0], m_v[1] * p[1], m_v[2] * p[2]); return np; };
		Point operator*(const double s) const { Point np(m_v[0] * s, m_v[1] * s, m_v[2] * s); return np; };
		Point operator/(const double s) const { Point np(m_v[0] / s, m_v[1] / s, m_v[2] / s); return np; };
		Point operator-() const { Point p(-m_v[0], -m_v[1], -m_v[2]); return p; };
		Point operator^(const Point & p) const {
			Point np(m_v[1] * p[2] - m_v[2] * p[1],
				  m_v[2] * p[0] - m_v[0] * p[2],
				  m_v[0] * p[1] - m_v[1] * p[0]);
			return np;
		};
		double operator *(const Point &p) const { return m_v[0] * p[0] + m_v[1] * p[1] + m_v[2] * p[2]; };
		
		bool operator == (const Point &p) const { return (m_v[0] == p[0] && m_v[1] == p[1] && m_v[2] == p[2]); };
		// bool operator <(const Point &p) const {}
		// angle between v and p
		double angle(Point &p) { return acos((*this) * p / (norm() * p.norm())); };
		double x(){ return m_v[0]; };
		double y(){ return m_v[1]; };
		double z(){ return m_v[2]; };

        Point rotate(double theta, Point vector);
	private:
		double m_v[3];
	};

	class Point2 {
	public:
		Point2(double x, double y) { m_v[0] = x; m_v[1] = y; };
		Point2(const Point2 &p) { m_v[0] = p[0]; m_v[1] = p[1]; };
		Point2(){ m_v[0] = m_v[1] = 0; };
		~Point2(){};

		double &operator[](int i)       { assert(0 <= i && i < 2); return m_v[i]; };
		double  operator()(int i) const { assert(0 <= i && i < 2); return m_v[i]; };
		double  operator[](int i) const { assert(0 <= i && i < 2); return m_v[i]; };
		bool operator == (const Point2 &p) { return (m_v[0] == p[0] && m_v[1] == p[1]); };
		// l2-norm
		double norm() const { return sqrt(m_v[0] * m_v[0] + m_v[1] * m_v[1]); };
		// lp-norm
		double norm(int i) const { return pow(pow(m_v[0], i) + pow(m_v[1], i), 1 / double(i)); };

		Point2 &operator += (const Point2 &p) { m_v[0] += p[0]; m_v[1] += p[1]; return *this; };
		Point2 &operator -= (const Point2 &p) { m_v[0] -= p[0]; m_v[1] -= p[1]; return *this; };
		Point2 &operator *= (const double s) { m_v[0] *= s;    m_v[1] *= s;    return *this; };
		Point2 &operator /= (const double s) { m_v[0] /= s;    m_v[1] /= s;    return *this; };

		Point2 operator+(const Point2 &p) const { Point2 np(m_v[0] + p[0], m_v[1] + p[1]); return np; };
		Point2 operator-(const Point2 &p) const { Point2 np(m_v[0] - p[0], m_v[1] - p[1]); return np; };
		// Point operator*(const Point &p) const { Point np(m_v[0] * p[0], m_v[1] * p[1]); return np; };
		Point2 operator*(const double s) const { Point2 np(m_v[0] * s, m_v[1] * s); return np; };
		Point2 operator/(const double s) const { Point2 np(m_v[0] / s, m_v[1] / s); return np; };
		Point2 operator-() const { Point2 p(-m_v[0], -m_v[1]); return p; };
		//Point2 operator^(const Point2 & p) const { };
		double operator *(const Point2 &p) const { return m_v[0] * p[0] + m_v[1] * p[1]; };

	private:
		double m_v[2]; 
	};
}

#endif
