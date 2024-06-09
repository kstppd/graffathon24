project('demo', ['cuda', 'cpp'],version : '0.4',default_options : ['warning_level=3', 'cpp_std=c++20'])


add_project_arguments(['--expt-relaxed-constexpr','-std=c++17','--expt-extended-lambda','-Xcompiler','-fopenmp' ], language: 'cuda')

inc_stb_image = include_directories('/home/kstppd/software/stb/',is_system:true)
inc_splitvector='/home/kstppd/dev/hashinator/include/splitvector'
inc_raylib='/home/kstppd/software/raylib/src/'
raylib = meson.get_compiler('cpp').find_library('raylib', required: true)


includes=[inc_stb_image,inc_splitvector,inc_raylib]
deps=[raylib]

executable('demo', 'demo.cu', include_directories:includes,link_args:'-fopenmp',dependencies:deps,install : true)



