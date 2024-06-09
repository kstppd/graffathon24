#pragma once
#include "raylib.h"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <limits>
#include <math.h>
#include <random>
#include <splitvec.h>
#include <stdlib.h>
#include <texture_types.h>
#include <vector3.h>

#define NUM_BANKS 32 // TODO depends on device
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

namespace Rmarch {
using u8 = uint8_t;
using u32 = uint32_t;
using u32 = uint32_t;
struct Pixel {
  u32 rgba;
  __host__ __device__ explicit Pixel() : rgba(0){};
  __host__ __device__ explicit Pixel(u32 a, u32 b, u32 g, u32 r) : rgba(0) {
    this->rgba |= (a & 255) << 24;
    this->rgba |= (b & 255) << 16;
    this->rgba |= (g & 255) << 8;
    this->rgba |= (r & 255) << 0;
  }
  __host__ __device__ u8 a() const { return (rgba & 0xFF000000) >> 24; }
  __host__ __device__ u8 b() const { return (rgba & 0x00FF0000) >> 16; }
  __host__ __device__ u8 g() const { return (rgba & 0x0000FF00) >> 8; }
  __host__ __device__ u8 r() const { return (rgba & 0x000000FF) >> 0; }

  __host__ __device__ void scale(float s) {
    rgba = ((static_cast<u32>(a()) & 255) << 24) |
           ((static_cast<u32>(b() * s) & 255) << 16) |
           ((static_cast<u32>(g() * s) & 255) << 8) |
           ((static_cast<u32>(r() * s) & 255) << 0);
  }

  __host__ __device__ Pixel &operator+=(const Pixel &other) {
    u32 newA = static_cast<int>(a()) + other.a();
    u32 newB = static_cast<int>(b()) + other.b();
    u32 newG = static_cast<int>(g()) + other.g();
    u32 newR = static_cast<int>(r()) + other.r();

    newA = clamp(newA, 0, 255);
    newB = clamp(newB, 0, 255);
    newG = clamp(newG, 0, 255);
    newR = clamp(newR, 0, 255);

    rgba = (static_cast<u32>(newA) & 255) << 24 |
           (static_cast<u32>(newB) & 255) << 16 |
           (static_cast<u32>(newG) & 255) << 8 |
           (static_cast<u32>(newR) & 255) << 0;
    return *this;
  }
};

__host__ __device__ void cart2sph(const Vector3<float> &cart, float &theta,
                                  float &phi) {
  float x = cart._x;
  float y = cart._y;
  float z = cart._z;
  phi = std::atan2(z, x);
  theta = std::acos(y / std::sqrt(x * x + y * y + z * z));
}

__host__ __device__ float legendre(int l, int m, float x) {
  float pmm = 1.0f;
  if (m > 0) {
    float somx2 = std::sqrt((1.0f - x) * (1.0f + x));
    float fact = 1.0f;
    for (int i = 1; i <= m; ++i) {
      pmm *= -fact * somx2;
      fact += 2.0f;
    }
  }

  if (l == m) {
    return pmm;
  }

  float pmmp1 = x * (2.0f * m + 1.0f) * pmm;
  if (l == m + 1) {
    return pmmp1;
  }

  float pll = 0.0f;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = ((2.0f * ll - 1.0f) * x * pmmp1 - (ll + m - 1.0f) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}

__host__ __device__ float sphHarm(int l, int m, float theta, float phi) {
  float sq = std::sqrt(2.0f);
  if (m == 0) {
    return legendre(l, 0.f, cos(theta));
  }
  if (m > 0) {
    return sq * std::cos(m * phi) * legendre(l, m, std::cos(theta));
  }

  return sq * std::sin(-m * phi) * legendre(l, -m, std::cos(theta));
}

__host__ __device__ float rand(Vector3<float> p) {
  return fmodf(sinf(dot(p, Vector3<float>{12.345, 67.89, 412.12})) * 42123.45f,
               1.0f) *
             2.0f -
         1.0f;
}

__host__ __device__ float mix(float x, float y, float a) {
  return x * (1.0f - a) + y * a;
}

__host__ __device__ float valueNoise(Rmarch::Vector3<float> p) {
  Rmarch::Vector3<float> u = floor(p);
  Rmarch::Vector3<float> v = p - u;
  Rmarch::Vector3<float> s =
      smoothstep(Rmarch::Vector3<float>(0.0), Rmarch::Vector3<float>(1.0), v);

  float a = rand(u);
  float b = rand(u + Rmarch::Vector3<float>(1.0, 0.0, 0.0));
  float c = rand(u + Rmarch::Vector3<float>(0.0, 1.0, 0.0));
  float d = rand(u + Rmarch::Vector3<float>(1.0, 1.0, 0.0));
  float e = rand(u + Rmarch::Vector3<float>(0.0, 0.0, 1.0));
  float f = rand(u + Rmarch::Vector3<float>(1.0, 0.0, 1.0));
  float g = rand(u + Rmarch::Vector3<float>(0.0, 1.0, 1.0));
  float h = rand(u + Rmarch::Vector3<float>(1.0, 1.0, 1.0));

  return mix(mix(mix(a, b, s._x), mix(c, d, s._x), s._y),
             mix(mix(e, f, s._x), mix(g, h, s._x), s._y), s._z);
}

__host__ __device__ float fbm(Rmarch::Vector3<float> p, float time) {
  Rmarch::Vector3<float> q = p - Rmarch::Vector3<float>(0.1, 0.0, 0.0) * 1.0;
  int numOctaves = 8;
  float weight = 0.5;
  float ret = 0.0;

  for (int i = 0; i < numOctaves; i++) {
    ret += weight * valueNoise(q);
    q *= 2.0;
    weight *= 0.5;
  }
  return clamp(ret - p._y, 0.0f, 1.0f);
}

struct SphereInternals {
  Vector3<float> origin;
  float rho;
  Pixel color;
  Pixel *map = nullptr;
  size_t map_h;
  size_t map_w;

  __host__ __device__ SphereInternals(Vector3<float> origin, float rho, Pixel c)
      : origin(origin), rho(rho), color(c) {}

  __host__ __device__ float sdf(const Vector3<float> &p) const noexcept {
    return (origin.distance(p) - rho);
  }

  __host__ __device__ float density_at(const Vector3<float> &p,
                                       float time) const noexcept {

    float sdf = origin.distance(p) - rho;
    return sdf > 0.0 ? 0 : std::abs(sdf);
  }

  __host__ __device__ Pixel color_at(const Vector3<float> &loc,
                                     float time) const noexcept {
    float tangle = time * 1.5f;
    float phi = std::atan2(loc._z, loc._x);
    float theta = (std::asin(loc._y));
    theta += 2 * tangle * M_PI / 180.0f;
    phi += tangle * M_PI / 180.0f;
    float u = (phi + M_PI) / (2.0f * M_PI);
    float v = (M_PI_2 - theta) / M_PI;
    int x = int(u * map_w) % map_w;
    int y = int(v * map_h) % map_h;
    x = std::min((size_t)std::max(x, 0), map_w - 1);
    y = std::min((size_t)std::max(y, 0), map_h - 1);
    Pixel p = map[y * map_w + x];
    return p;
  }

  __host__ __device__ Vector3<float>
  get_normal_at(const Vector3<float> &position, float ds = 0.1) const noexcept {
    const auto xfwd = add(position, Vector3<float>{ds, 0.0, 0.0});
    const auto xbwd = add(position, Vector3<float>{-ds, 0.0, 0.0});
    const auto yfwd = add(position, Vector3<float>{0.0, ds, 0.0});
    const auto ybwd = add(position, Vector3<float>{0.0, -ds, 0.0});
    const auto zfwd = add(position, Vector3<float>{0.0, 0.0, ds});
    const auto zbwd = add(position, Vector3<float>{0.0, 0.0, -ds});

    Vector3<float> normal{
        this->sdf(xfwd) - this->sdf(xbwd),
        this->sdf(yfwd) - this->sdf(ybwd),
        this->sdf(zfwd) - this->sdf(zbwd),
    };
    normal.normalize();
    return normal;
  }
};

struct EllipsoidInternals {
  Vector3<float> state;
  Vector3<float> origin;
  Pixel color;
  float dtheta, dphi;

  __host__ __device__ EllipsoidInternals(Vector3<float> state,
                                         Vector3<float> origin, Pixel c)
      : state(state), origin(origin), color(c) {}

  __host__ __device__ float sdf(const Vector3<float> &p) const noexcept {
    // return (origin.distance(p) - rho);
    auto shifted = sub(p, origin);
    float k0 = magnitude(div(shifted, state));
    float k1 = magnitude(div(shifted, (mul(state, state))));
    return k0 * (k0 - 1.0) / k1;
  }

  __host__ __device__ Pixel color_at(const Vector3<float> &loc,
                                     float time) const noexcept {
    return color;
  }

  __host__ __device__ void modulate_color(float time, float pa, float pb,
                                          float pc) noexcept {
    float f = 1;
    u32 r = 255.0 * (1.0+std::sin(2.0 * M_PI * f * time + pa))/2.0;
    u32 g = 255.0 * (1.0+std::sin(2.0 * M_PI * f * time + pb))/2.0;
    u32 b = 255.0 * (1.0+std::sin(2.0 * M_PI * f * time + pc))/2.0;
    color = Pixel{255, b, g, r};
  }

  __host__ __device__ Vector3<float>
  get_normal_at(const Vector3<float> &position, float ds = 0.1) const noexcept {
    const auto xfwd = add(position, Vector3<float>{ds, 0.0, 0.0});
    const auto xbwd = add(position, Vector3<float>{-ds, 0.0, 0.0});
    const auto yfwd = add(position, Vector3<float>{0.0, ds, 0.0});
    const auto ybwd = add(position, Vector3<float>{0.0, -ds, 0.0});
    const auto zfwd = add(position, Vector3<float>{0.0, 0.0, ds});
    const auto zbwd = add(position, Vector3<float>{0.0, 0.0, -ds});

    Vector3<float> normal{
        this->sdf(xfwd) - this->sdf(xbwd),
        this->sdf(yfwd) - this->sdf(ybwd),
        this->sdf(zfwd) - this->sdf(zbwd),
    };
    normal.normalize();
    return normal;
  }
};

enum ItemKind { S, E };
struct Item {
  __host__ __device__ Item() {}

  ItemKind kind;
  union {
    SphereInternals sphere;
    EllipsoidInternals ellipsoid;
  };

  __host__ __device__ float sdf(Vector3<float> p) const noexcept {
    switch (kind) {
    case (ItemKind::S):
      return sphere.sdf(p);
    case (ItemKind::E):
      return ellipsoid.sdf(p);
    }
  }

  __host__ __device__ Vector3<float>
  normal_at(Vector3<float> p) const noexcept {
    switch (kind) {
    case (ItemKind::S):
      return sphere.get_normal_at(p);
    case (ItemKind::E):
      return ellipsoid.get_normal_at(p);
    }
  }

  __host__ __device__ Pixel color(const Vector3<float> &loc,
                                  float time) const noexcept {
    // return sphere.color_at(loc, time);
    switch (kind) {
    case (ItemKind::S):
      return sphere.color_at(loc, time);
    case (ItemKind::E):
      return ellipsoid.color_at(loc, time);
    }
  }

  __host__ __device__ float density(const Vector3<float> &loc,
                                    float time) const noexcept {
    // return sphere.density_at(loc, time);
  }
};

struct Light {
  float dtheta = 1.5;
  float dphi = 1.5;
  Vector3<float> _origin, _direction;

  explicit Light(Vector3<float> o, Vector3<float> d)
      : _origin(o), _direction(d) {}

  explicit Light()
      : _origin(Vector3<float>{0.0}), _direction(Vector3<float>{0.0}) {}
};

template <int W, int H, int MAX_DIST>
__global__ void
render(Pixel *__restrict__ pixels, Item *const __restrict__ objects,
       const Light *const lights, const size_t n_pixels, const size_t n_objects,
       const size_t n_lighs, Vector3<float> camera_pos,
       Vector3<float> camera_target, float time,float mod_freq) {

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const float fov = std::tan(30. * (M_PI / 180.) * 0.5f);
  constexpr float aspect = static_cast<float>(W) / static_cast<float>(H);
  const size_t i = tid % W;
  const size_t j = tid / W;
  const float x = (2 * (i + 0.5f) / W - 1) * (aspect)*fov;
  const float y = (1 - 2 * (j + 0.5f) / H) * fov;

  Vector3<float> ray_dir = Vector3<float>(x, y, 0);
  ray_dir._z =
      std::sqrt(1. - ray_dir._x * ray_dir._x - ray_dir._y * ray_dir._y);
  ray_dir.normalize();
  Vector3<float> ray_pos = camera_pos;

  float min_dist = std::numeric_limits<float>::max();
  size_t hit_id = 0;
  float travel = 0.0;
  while (travel < MAX_DIST) {
    for (size_t i = 0; i < n_objects; ++i) {
      const auto d = objects[i].sdf(ray_pos);
      if (d < min_dist) {
        min_dist = d;
        hit_id = i;
      }
    }

    auto modulator=[&]()->float{
        return std::cos(2.0*M_PI*mod_freq*time);
      
    };

    // Hacky way to get the hit without slowing down too much
    if (min_dist <= 1e-2) {
      Pixel rgba = objects[hit_id].color(ray_pos, time);
      const auto normal = objects[hit_id].normal_at(ray_pos);
      auto toLight = sub(lights[0]._origin, ray_pos);
      toLight.normalize();
      float diffuse = std::max(0.0f, dot(normal, toLight));
      rgba.scale(diffuse*modulator());
      pixels[tid] = rgba;
      break;
    }

    // Advance
    ray_pos._x += ray_dir._x * min_dist;
    ray_pos._y += ray_dir._y * min_dist;
    ray_pos._z += ray_dir._z * min_dist;
    travel += min_dist;
  }
}

template <int W, int H, int MAX_STEPS>
__global__ void render_volume(Pixel *__restrict__ pixels,
                              Item *const __restrict__ objects,
                              const Light *const lights, const size_t n_pixels,
                              const size_t n_objects, const size_t n_lighs,
                              Vector3<float> camera_pos,
                              Vector3<float> camera_target, float time) {

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const float fov = std::tan(30. * (M_PI / 180.) * 0.5f);
  constexpr float aspect = static_cast<float>(W) / static_cast<float>(H);
  const size_t i = tid % W;
  const size_t j = tid / W;
  const float x = (2 * (i + 0.5f) / W - 1) * (aspect)*fov;
  const float y = (1 - 2 * (j + 0.5f) / H) * fov;

  if (tid >= n_pixels) {
    return;
  }

  Vector3<float> ray_dir = Vector3<float>(x, y, 0);
  ray_dir._z =
      std::sqrt(1. - ray_dir._x * ray_dir._x - ray_dir._y * ray_dir._y);
  ray_dir.normalize();
  Vector3<float> ray_pos = camera_pos;

  constexpr float ds = 0.1;
  size_t steps = 0;
  float total_dens = 0.0;
  while (steps < MAX_STEPS) {
    for (size_t i = 0; i < n_objects; ++i) {
      const auto density = objects[i].density(ray_pos, time);
      if (density > 0.0) {
        total_dens += density;
      }
    }

    // Advance
    ray_pos._x += ray_dir._x * ds;
    ray_pos._y += ray_dir._y * ds;
    ray_pos._z += ray_dir._z * ds;
    steps++;
  }
  if (total_dens > 0) {
    float c = 255 - clamp(255 * (1. - total_dens) * (1. + total_dens), 0, 255);
    pixels[tid] = Pixel(c, c, c, c);
  }
}

} // namespace Rmarch
