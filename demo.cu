#if 0 
nvcc demo.cu libraylib.a  -O3  -I.  -I./hashinator/include/splitvector --expt-relaxed-constexpr -std=c++17 --expt-extended-lambda -o demo
exit(0)
#endif
#include "vector3.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <linux/limits.h>
#include <random>
#include <raylib.h>
#include <rmarch.hpp>
#include <vector>

constexpr size_t H = 1080;
constexpr size_t W = 1920;
constexpr size_t threads = 1024;
// #define DEG2RAD  M_PI / 180.0
// #define RAD2DEG  180.0 / M_PI
size_t satellites = 0;
constexpr size_t max_satellites = 16;
constexpr float sat_period = 1.0;
static bool init_phases = false;
static bool init_music = false;
float p_a[max_satellites];
float p_b[max_satellites];
float p_c[max_satellites];

template <typename T>
using splitvector = split::SplitVector<T, split::split_unified_allocator<T>>;
using namespace Rmarch;
using u8 = uint8_t;
using u32 = uint32_t;
using type_t = float;

splitvector<Item> *scene;
splitvector<Light> *lights;
Pixel *gpu_image_buffer;
bool scene_01_init = false;
bool scene_02_init = false;
bool scene_03_init = false;

Rmarch::Vector3<float> LorentzAttractor(const Rmarch::Vector3<float> &location,
                                        float dt) {
  float rho = 28.0;
  float beta = 8. / 3.0;
  float sigma = 10.0;
  float dx = sigma * (location._y - location._x) * dt;
  float dy = (location._x * (rho - location._z) - location._y) * dt;
  float dz = (location._x * location._y - beta * location._z) * dt;
  return Rmarch::Vector3<float>{location._x + dx, location._y + dy,
                                location._z + dz};
}

float getRandomFloat(float min, float max) {
  return min + static_cast<float>(rand()) /
                   (static_cast<float>(RAND_MAX) / (max - min));
}

float fwrap(float x, float min, float max) {
  if (min > max) {
    return fwrap(x, max, min);
  }
  return (x >= 0 ? min : max) + std::fmod(x, max - min);
}

Rmarch::Vector3<float> nextOrbitalPoint(Rmarch::Vector3<float> x0,
                                        float &dtheta, float &dphi) {

  float r = std::sqrt(x0._x * x0._x + x0._y * x0._y + x0._z * x0._z);
  float theta = std::acos(x0._y / r) * RAD2DEG;
  float phi = std::atan2(x0._z, x0._x) * RAD2DEG;

  float theta_new = (theta + dtheta);
  float phi_new = (phi + dphi);
  if (theta_new > 179.0) {
    dtheta *= -1;
  }

  if (theta_new < 1.0) {
    dtheta *= -1;
  }
  fwrap(theta_new, 0.0, 179.0);
  fwrap(phi_new, -179.0, 179.0);
  theta_new *= DEG2RAD;
  phi_new *= DEG2RAD;

  Rmarch::Vector3<float> x1;
  x1._x = r * std::sin(theta_new) * std::cos(phi_new);
  x1._z = r * std::sin(theta_new) * std::sin(phi_new);
  x1._y = r * std::cos(theta_new);
  return x1;
}

void scene_01(Image *canvas, float t0, cudaStream_t s) {

  size_t pixels = W * H;
  if (!scene_01_init) {
    scene_01_init = true;
    Image earth = LoadImage("earth.png");
    Item a;
    a.sphere = SphereInternals(Rmarch::Vector3<float>{0, 0, 0}, 1.0,
                               Pixel(255, 255, 0, 0));

    cudaMalloc(&(a.sphere.map), sizeof(Pixel) * earth.height * earth.width);
    assert(a.sphere.map && "ERROR:Could not allocate earth texture map");
    cudaMemcpy(a.sphere.map, earth.data,
               sizeof(Pixel) * earth.height * earth.width,
               cudaMemcpyHostToDevice);
    a.sphere.map_h = earth.height;
    a.sphere.map_w = earth.width;
    a.kind = ItemKind::S;

    scene->push_back(std::move(a));
    lights->push_back(Light{Rmarch::Vector3<float>{0., 1., -2.},
                            Rmarch::Vector3<float>{-1., 0., 0.}});
  }

  static float z0 = -1.2;
  static float x0 = -0.0;
  static float y0 = 0.0;
  float time = GetTime();
  float z = std::max(z0 - 0.2 * (time - t0), -5.0);
  // float y = std::min(y0 + 0.2 * (time - t0), 0.75);
  float y = std::sin(2.0 * M_PI * (time - t0) / t0);

  constexpr int ss = 4;
  const size_t blocks = pixels / threads;

  float mod = 0.0;
  if (time > 24.0) {
    mod = 0.75;
  }
  if (time > 28.0) {
    mod = 1.0;
  }
  render<W, H, 10><<<ss * blocks, threads / ss, 0, s>>>(
      gpu_image_buffer, scene->data(), lights->data(), pixels, scene->size(),
      lights->size(), Rmarch::Vector3<float>{x0, y0, z},
      Rmarch::Vector3<float>{0.0, 0.0, 0.0}, time, mod);
}

void scene_02(Image *canvas, float t0, cudaStream_t s, bool add_one = false) {

  size_t pixels = W * H;
  if (!scene_02_init) {
    scene->reserve(20);
  }

  if (add_one || !scene_02_init) {

    if (!scene_02_init) {
      scene_02_init = true;
    }
    Item b;
    const EllipsoidInternals &last_sat = (*scene)[satellites].ellipsoid;
    int _r = rand() % 255;
    int _g = rand() % 255;
    int _b = rand() % 255;
    b.ellipsoid = EllipsoidInternals(
        Rmarch::Vector3<float>{0.75 / 10, 0.25 / 10, 0.75 / 10},
        Rmarch::Vector3<float>{0.0, 0.2, -1.2}, Pixel(255, _b, _g, _r));

    b.ellipsoid.dtheta = 2.0;
    b.ellipsoid.dphi = 1.0;
    b.kind = ItemKind::E;
    scene->push_back(std::move(b));
    satellites++;
  }

  float time = GetTime();
  float z = -5.0;
  float y = 0.0;
  float dt = time - t0;

  auto tt = GetFrameTime();
  if (satellites == max_satellites) {
    Light &l = (*lights)[0];
    auto new_pos = nextOrbitalPoint(l._origin, l.dtheta, l.dphi);
    l._origin = new_pos;
  }

  for (int i = 1; i <= satellites + 1; i++) {
    EllipsoidInternals &ell = (*scene)[i].ellipsoid;
    auto new_pos = nextOrbitalPoint(ell.origin, ell.dtheta, ell.dphi);
    ell.origin = new_pos;
  }

  Light &l = (*lights)[0];
  auto new_pos = nextOrbitalPoint(l._origin, l.dtheta, l.dphi);
  l._origin = new_pos;

  scene->optimizeGPU(s);
  constexpr int ss = 4;
  const size_t blocks = pixels / threads;

  render<W, H, 8><<<ss * blocks, threads / ss, 0, s>>>(
      gpu_image_buffer, scene->data(), lights->data(), pixels, scene->size(),
      lights->size(), Rmarch::Vector3<float>{0, y, z},
      Rmarch::Vector3<float>{0.0, 0.0, 0.0}, time, 0.0);
}

void scene_03(Image *canvas, float t0, cudaStream_t s, bool add_one = false) {

  size_t pixels = W * H;
  float time = GetTime();
  float z = -5.0;
  float y = 0.0;
  float dt = time - t0;

  // if (!scene_03_init){

  //   bool scene_03_init = true;
  //   Light &l = (*lights)[0];
  //   l.dtheta*=4.0;
  //   l.dtheta*=4.0;
  // }

  if (!init_phases) {
    for (int i = 0; i < satellites; i++) {
      p_a[i] = getRandomFloat(0.0, 360) * DEG2RAD;
      p_b[i] = getRandomFloat(0.0, 360) * DEG2RAD;
      p_c[i] = getRandomFloat(0.0, 360) * DEG2RAD;
    }
    init_phases = true;
  }

  for (int i = 1; i <= satellites + 1; i++) {
    EllipsoidInternals &ell = (*scene)[i].ellipsoid;
    auto new_pos = nextOrbitalPoint(ell.origin, ell.dtheta, ell.dphi);
    ell.modulate_color(time - t0, p_a[i - 1], p_b[i - 1], p_c[i - 1]);
    ell.origin = new_pos;
  }

  Light &l = (*lights)[0];
  // auto new_pos = nextOrbitalPoint(l._origin, l.dtheta, l.dphi);
  l._origin = Rmarch::Vector3<float>{0., 1.0, -5.0};

  scene->optimizeGPU(s);
  constexpr int ss = 4;
  const size_t blocks = pixels / threads;

  render<W, H, 8><<<ss * blocks, threads / ss, 0, s>>>(
      gpu_image_buffer, scene->data(), lights->data(), pixels, scene->size(),
      lights->size(), Rmarch::Vector3<float>{0, y, z},
      Rmarch::Vector3<float>{0.0, 0.0, 0.0}, time, 0.0);
}

float scene_04(Image *canvas, float t0, cudaStream_t s, bool add_one = false,
               float acc = 100.0, float z0 = -5.0) {

  size_t pixels = W * H;
  float time = GetTime();
  float y0 = 0.0;
  float dt = time - t0;

  float v0 = 0.0;
  float vel = v0 - 0.5 * dt * acc;
  float z = z0 + dt * vel;
  float y = 0.0;
  ;

  for (int i = 1; i <= satellites + 1; i++) {
    EllipsoidInternals &ell = (*scene)[i].ellipsoid;
    auto new_pos = nextOrbitalPoint(ell.origin, ell.dtheta, ell.dphi);
    ell.modulate_color(time - t0, p_a[i - 1], p_b[i - 1], p_c[i - 1]);
    ell.origin = new_pos;
  }

  scene->optimizeGPU(s);
  constexpr int ss = 4;
  const size_t blocks = pixels / threads;

  render<W, H, 8><<<ss * blocks, threads / ss, 0, s>>>(
      gpu_image_buffer, scene->data(), lights->data(), pixels, scene->size(),
      lights->size(), Rmarch::Vector3<float>{0, y, z},
      Rmarch::Vector3<float>{0.0, 0.0, 0.0}, time, 0.0);
  return z;
}

int main() {

  srand(time(nullptr));
  int display = GetCurrentMonitor();
  float time = 0.0;
  size_t pixels = W * H;
  std::vector<Pixel> cpu_image_buffer(pixels, Pixel(255, 0, 0, 255));
  cudaStream_t video_stream;
  cudaStreamCreate(&video_stream);

  Image bg = LoadImage("stars.png");
  ImageResize(&bg, W, H);
  memcpy(cpu_image_buffer.data(), bg.data, sizeof(Pixel) * W * H);
  cudaMalloc(&gpu_image_buffer, pixels * sizeof(Pixel));
  assert(gpu_image_buffer && "ERROR: failed to malloc gpu_image_buffer");
  cudaMemcpy(gpu_image_buffer, cpu_image_buffer.data(), pixels * sizeof(Pixel),
             cudaMemcpyHostToDevice);

  Image img = {.data = cpu_image_buffer.data(),
               .width = W,
               .height = H,
               .mipmaps = 1,
               .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};

  scene = new splitvector<Item>();
  lights = new splitvector<Light>();
  InitWindow(W, H, "Demo");
  SetWindowState(FLAG_WINDOW_RESIZABLE);
  SetExitKey(KEY_ESCAPE);
  SetTargetFPS(60);
  ToggleFullscreen();
  SetWindowSize(W, H);

  Texture2D texture = LoadTextureFromImage(img);
  InitAudioDevice();
  auto music = LoadSound("music.wav");
  bool skip = false;
  float sat_time = -1.0;
  // DisableCursor();
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    memset(cpu_image_buffer.data(), 0, pixels * sizeof(Pixel));

    if (time < 2.0) {
      const char *hello_message = "Hello Graffathon!";
      int text_width = MeasureText(hello_message, 100);
      ImageDrawText(&img, hello_message, W / 2 - 0.5 * text_width, H / 2, 100,
                    ORANGE);
    }

    if (time > 2.0 && time < 4.0) {
      const char *message = "Ready for a journey in space...?";
      int text_width = MeasureText(message, 100);
      ImageDrawText(&img, message, W / 2 - 0.5 * text_width, H / 2, 100,
                    ORANGE);
    }

    if (time > 4.0 && time < 32) {
      if (!init_music) {
        init_music = true;
        PlaySound(music);
      }
      cudaMemcpyAsync(gpu_image_buffer, bg.data, pixels * sizeof(Pixel),
                      cudaMemcpyHostToDevice, video_stream);
      scene_01(&img, 4.0, video_stream);
      cudaMemcpyAsync(cpu_image_buffer.data(), gpu_image_buffer,
                      pixels * sizeof(Pixel), cudaMemcpyDeviceToHost,
                      video_stream);
      cudaStreamSynchronize(video_stream);
    }

    if (time > 32.0 && time < 64) {
      sat_time += GetFrameTime();

      bool addsat = false;
      if (sat_time > sat_period && satellites < max_satellites) {
        addsat = true;
        sat_time = 0.0;
      }
      cudaMemcpyAsync(gpu_image_buffer, bg.data, pixels * sizeof(Pixel),
                      cudaMemcpyHostToDevice, video_stream);
      scene_02(&img, 32.0, video_stream, addsat);
      cudaStreamSynchronize(video_stream);
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cpu_image_buffer.data(), gpu_image_buffer,
                      pixels * sizeof(Pixel), cudaMemcpyDeviceToHost,
                      video_stream);
      cudaStreamSynchronize(video_stream);
    }

    if (time > 68 && time < 84) {
      cudaMemcpyAsync(gpu_image_buffer, bg.data, pixels * sizeof(Pixel),
                      cudaMemcpyHostToDevice, video_stream);
      scene_03(&img, 68., video_stream, false);
      cudaStreamSynchronize(video_stream);
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cpu_image_buffer.data(), gpu_image_buffer,
                      pixels * sizeof(Pixel), cudaMemcpyDeviceToHost,
                      video_stream);
      cudaStreamSynchronize(video_stream);
    }

    float store_z;
    if (time > 84 && time < 90) {
      cudaMemcpyAsync(gpu_image_buffer, bg.data, pixels * sizeof(Pixel),
                      cudaMemcpyHostToDevice, video_stream);
      store_z = scene_04(&img, 84., video_stream, false, 8.0 / 100.0);
      cudaStreamSynchronize(video_stream);
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cpu_image_buffer.data(), gpu_image_buffer,
                      pixels * sizeof(Pixel), cudaMemcpyDeviceToHost,
                      video_stream);
      cudaStreamSynchronize(video_stream);
    }

    if (time > 90 && time < 105) {
      cudaMemcpyAsync(gpu_image_buffer, bg.data, pixels * sizeof(Pixel),
                      cudaMemcpyHostToDevice, video_stream);
      scene_04(&img, 90., video_stream, false, -10.0 / 100.0, store_z);
      cudaStreamSynchronize(video_stream);
      cudaDeviceSynchronize();
      cudaMemcpyAsync(cpu_image_buffer.data(), gpu_image_buffer,
                      pixels * sizeof(Pixel), cudaMemcpyDeviceToHost,
                      video_stream);
      cudaStreamSynchronize(video_stream);
    }
    time = GetTime();

    UpdateTexture(texture, img.data);
    DrawTexture(texture, 0, 0, WHITE);
    DrawFPS(0, 0);
    if (time > 105) {
      break;
    }
    EndDrawing();
  }
  CloseWindow();

  return 0;
}
