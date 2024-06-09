#pragma once
#include <algorithm>

namespace Rmarch {
template <typename T> struct Vector3 {
  T _x;
  T _y;
  T _z;
  // T _w=0.0; //to get 128 but alignment

  __host__ __device__ explicit Vector3() : _x(T(0)), _y(T(0)), _z(T(0)) {}

  __host__ __device__ explicit Vector3(T val) {
    _x = val;
    _y = val;
    _z = val;
  }

  __host__ __device__ explicit Vector3(T x, T y, T z) {
    _x = x;
    _y = y;
    _z = z;
  }

  __host__ __device__ float distance(const Vector3<T> &other) const {
    Vector3<T> d{other._x - _x, other._y - _y, other._z - _z};
    return magnitude(d);
  }

  __host__ __device__ float distance(const float3 &other) const {
    Vector3<T> d{other.x - _x, other.y - _y, other.z - _z};
    return magnitude(d);
  }

  __host__ __device__ void normalize() noexcept {
    float mag = magnitude(*this);
    _x /= mag;
    _y /= mag;
    _z /= mag;
  }

  __host__ __device__ void operator+=(const Vector3<T> other) {
    _x += other._x;
    _y += other._y;
    _z += other._z;
  }
  __host__ __device__ void operator-=(const Vector3<T> other) {
    _x -= other._x;
    _y -= other._y;
    _z -= other._z;
  }
  __host__ __device__ void operator+=(float s) {
    _x += s;
    _y += s;
    _z += s;
  }
  __host__ __device__ void operator-=(float s) {
    _x -= s;
    _y -= s;
    _z -= s;
  }
  __host__ __device__ void operator*=(float s) {
    _x *= s;
    _y *= s;
    _z *= s;
  }
  __host__ __device__ void operator/=(float s) {
    _x /= s;
    _y /= s;
    _z /= s;
  }
};

template <typename T>
__host__ __device__ float magnitude(const Vector3<T> &vec) {
  return std::sqrt(vec._x * vec._x + vec._y * vec._y + vec._z * vec._z);
}

template <typename T>
__host__ __device__ Vector3<T> unit(const Vector3<T> &vec) {
  float mag = magnitude(vec);
  return Vector3<T>{vec._x / mag, vec._y / mag, vec._z / mag};
}

template <typename T>
__host__ __device__ Vector3<T> add(const Vector3<T> &lhs,
                                   const Vector3<T> &rhs) {
  return Vector3<T>{lhs._x + rhs._x, lhs._y + rhs._y, lhs._z + rhs._z};
}

template <typename T>
__host__ __device__ Vector3<T> div(const Vector3<T> &lhs,
                                   const Vector3<T> &rhs) {
  return Vector3<T>{lhs._x / rhs._x, lhs._y / rhs._y, lhs._z / rhs._z};
}

template <typename T>
__host__ __device__ Vector3<T> mul(const Vector3<T> &lhs,
                                   const Vector3<T> &rhs) {
  return Vector3<T>{lhs._x * rhs._x, lhs._y * rhs._y, lhs._z * rhs._z};
}

template <typename T>
__host__ __device__ Vector3<T> mul(const T s, const Vector3<T> &v) {
  return mul(v, s);
}

template <typename T>
__host__ __device__ Vector3<T> mul(const Vector3<T> &v, const T s) {
  return mul(v, s);
}

template <typename T> __host__ __device__ int sign(T val) {
  return (T(0) < val) - (val < T(0));
}

template <typename T>
__host__ __device__ Vector3<T> sub(const Vector3<T> &lhs,
                                   const Vector3<T> &rhs) {
  return Vector3<T>{lhs._x + rhs._x, lhs._y + rhs._y, lhs._z + rhs._z};
}

template <typename T>
__host__ __device__ Vector3<T> sub(const Vector3<T> &v, const T &scalar) {
  return Vector3<T>(v._x - scalar, v._y - scalar, v._z - scalar);
}

template <typename T>
__host__ __device__ Vector3<T> sub(const T &scalar, const Vector3<T> &v) {
  return Vector3<T>(scalar - v._x, scalar - v._y, scalar - v._z);
}

template <typename T>
__host__ __device__ float dot(const Vector3<T> &lhs, const Vector3<T> &rhs) {
  return lhs._x * rhs._x + lhs._y * rhs._y + lhs._z * rhs._z;
}

template <typename T>
__host__ __device__ Vector3<T> cross(const Vector3<T> &lhs,
                                     const Vector3<T> &rhs) {
  return Vector3<T>{lhs._y * rhs._z - lhs._z * rhs._y,
                    lhs._z * rhs._x - lhs._x * rhs._z,
                    lhs._x * rhs._y - lhs._y * rhs._x};
}

template <typename T> __host__ __device__ float dot2(Vector3<T> v) noexcept {
  return dot(v, v);
}

inline __host__ __device__ float clamp(float n, float lower, float upper) {
  return std::max(lower, std::min(n, upper));
}

template <typename T>
__host__ __device__ Vector3<T> max3(const Vector3<T> p, float S) {
  return Vector3<T>{(p.x >= S) ? p._x : S, (p.y >= S) ? p._y : S,
                    (p.z >= S) ? p._z : S};
}

template <typename T> __host__ __device__ float max3(const Vector3<T> &p) {
  return std::max(std::max(p._x, p._y), p._z);
}

template <typename T> __host__ __device__ Vector3<T> abs3(const Vector3<T> &p) {
  return Vector3<T>{std::abs(p._x), std::abs(p._y), std::abs(p._z)};
}

template <typename T>
__host__ __device__ Vector3<T> operator+(const Vector3<T> &a,
                                         const Vector3<T> &b) {
  Vector3<T> c(a);
  c += b;
  return c;
}

template <typename T>
__host__ __device__ Vector3<T> operator-(const Vector3<T> &a,
                                         const Vector3<T> &b) {
  Vector3<T> c(a);
  c -= b;
  return c;
}

template <typename T>
__host__ __device__ Vector3<T> operator*(Vector3<T> p, float S) {
  Vector3<T> retval(p);
  retval *= S;
  return retval;
}

template <typename T>
__host__ __device__ Vector3<T> clamp(const Vector3<T> &x, const T &minVal,
                                     const T &maxVal) {
  return Vector3<T>(clamp(x._x, minVal, maxVal), clamp(x._y, minVal, maxVal),
                    clamp(x._z, minVal, maxVal));
}

template <typename T>
__host__ __device__ Vector3<T>
smoothstep(const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &x) {
  Vector3<T> t = clamp(div((sub(x, a)), (sub(b, a))), 0.0f, 1.0f);
  // return t * t * (3.0 - 2.0 * t);
  return add(mul(mul(t, t), sub(T(3), mul(T(2), t))), mul(t, t));
}

template <typename T>
__host__ __device__ Vector3<T> floor(const Vector3<T> &v) {
  return Vector3<T>(std::floor(v._x), std::floor(v._y), std::floor(v._z));
}

} // namespace Rmarch
