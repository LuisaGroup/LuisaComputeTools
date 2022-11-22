includes("tinyexr")

_config_project({
    project_name = "lc-tools",
    project_kind = "shared"
})
local add_includedirs = _get_add_includedirs()
local add_defines = _get_add_defines()
add_deps("lc-runtime", "lc-dsl", "lc-vstl", "tinyexr")
add_defines("LC_TOOL_EXPORT_DLL")
add_files("src/**.cpp", "../ext/stb/stb.c")
add_includedirs("../ext/stb/")
