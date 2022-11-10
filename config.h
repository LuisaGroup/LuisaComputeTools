#pragma once
#ifdef LC_TOOL_EXPORT_DLL
#define LC_TOOL_API __declspec(dllexport)
#else
#define LC_TOOL_API __declspec(dllimport)
#endif