BuildProject({
    projectName = "lc-tools",
    projectType = "shared"
})
add_deps("lc-runtime", "lc-dsl", "lc-vstl")
lc_add_defines("LC_TOOL_EXPORT_DLL")
add_files("**.cpp", "../ext/stb/stb.c")
add_includedirs("../ext/stb/")
