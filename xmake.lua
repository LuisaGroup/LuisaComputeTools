add_subdirs("tinyexr")

BuildProject({
    projectName = "lc-tools",
    projectType = "shared"
})
add_deps("lc-runtime", "lc-dsl", "lc-vstl", "tinyexr")
lc_add_defines("LC_TOOL_EXPORT_DLL")
add_files("src/**.cpp", "../ext/stb/stb.c")
add_includedirs("../ext/stb/")
