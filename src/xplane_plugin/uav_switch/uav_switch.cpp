// UavSwitch: changes the AI aircraft model at runtime, driven by a text file.
//
// Python writes   <X-Plane>/Output/uav_switch/request.txt   as:  "<id>\t<acf path>"
// The plugin loads the model into AI slot 1 (first AI aircraft) and writes
//                 <X-Plane>/Output/uav_switch/ack.txt       as:  "<id>\t<ok|fail>\t<loaded path>"
#include "XPLMPlanes.h"
#include "XPLMProcessing.h"
#include "XPLMUtilities.h"
#include "XPLMPlugin.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static std::string g_dir;
static std::string g_last_id;
static bool g_acquired = false;

static std::string slurp(const std::string &path) {
    std::ifstream f(path.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void write_ack(const std::string &id, const char *status, const std::string &loaded) {
    std::string tmp = g_dir + "ack.tmp", fin = g_dir + "ack.txt";
    {
        std::ofstream f(tmp.c_str());
        f << id << "\t" << status << "\t" << loaded << "\n";
    }
    std::rename(tmp.c_str(), fin.c_str());
}

static float loop_cb(float, float, int, void *) {
    std::string req = slurp(g_dir + "request.txt");
    size_t tab = req.find('\t');
    if (tab == std::string::npos) return 0.5f;
    std::string id = req.substr(0, tab);
    std::string path = req.substr(tab + 1);
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) path.pop_back();
    if (id == g_last_id || path.empty()) return 0.5f;
    g_last_id = id;

    if (!g_acquired) {
        char *none[] = {NULL};
        g_acquired = XPLMAcquirePlanes(none, NULL, NULL) == 1;
    }
    if (!g_acquired) {
        write_ack(id, "fail", "could not acquire planes (another plugin owns them)");
        return 0.5f;
    }
    XPLMSetAircraftModel(1, path.c_str());
    XPLMDisableAIForPlane(1);   // AI plane stays where Python puts it

    char file[256] = {0}, loaded[512] = {0};
    XPLMGetNthAircraftModel(1, file, loaded);
    write_ack(id, "ok", loaded);
    return 0.5f;
}

PLUGIN_API int XPluginStart(char *name, char *sig, char *desc) {
    std::strcpy(name, "UavSwitch");
    std::strcpy(sig, "arikovani.uavswitch");
    std::strcpy(desc, "Loads AI aircraft models on request (file based).");
    XPLMEnableFeature("XPLM_USE_NATIVE_PATHS", 1);
    char sys[512] = {0};
    XPLMGetSystemPath(sys);
    g_dir = std::string(sys) + "Output/uav_switch/";
    return 1;
}

PLUGIN_API void XPluginStop(void) {}
PLUGIN_API int XPluginEnable(void) {
    XPLMRegisterFlightLoopCallback(loop_cb, 0.5f, NULL);
    return 1;
}
PLUGIN_API void XPluginDisable(void) {
    XPLMUnregisterFlightLoopCallback(loop_cb, NULL);
    if (g_acquired) { XPLMReleasePlanes(); g_acquired = false; }
}
PLUGIN_API void XPluginReceiveMessage(XPLMPluginID, int, void *) {}
