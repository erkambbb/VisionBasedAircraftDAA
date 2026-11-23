// UavSwitch: changes the AI aircraft model at runtime, driven by a text file.
//
// Python writes   <X-Plane>/Output/uav_switch/request.txt   as:  "<id>\t<acf path>[\t<slot>]"
// The plugin loads the model into AI slot <slot> (default 1 = first AI aircraft) and writes
//                 <X-Plane>/Output/uav_switch/ack.txt       as:  "<id>\t<ok|fail>\t<loaded path>"
//
// Terrain heights: Python writes probe_request.txt as "<id>" then one "<lat> <lon>" per line.
// The plugin answers probe_ack.tmp -> probe_ack.txt: "<id>" then one terrain height (m MSL) or "nan"
// per line. Only scenery X-Plane has loaded (around the user aircraft) can be probed.
#include "XPLMPlanes.h"
#include "XPLMProcessing.h"
#include "XPLMUtilities.h"
#include "XPLMPlugin.h"
#include "XPLMScenery.h"
#include "XPLMGraphics.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static std::string g_dir;
static std::string g_last_id;
static bool g_acquired = false;
static std::string g_last_probe;
static XPLMProbeRef g_probe = NULL;

static void handle_probe() {
    std::ifstream f((g_dir + "probe_request.txt").c_str());
    std::string id;
    if (!std::getline(f, id) || id.empty() || id == g_last_probe) return;
    g_last_probe = id;
    if (!g_probe) g_probe = XPLMCreateProbe(xplm_ProbeY);
    std::ostringstream out;
    out << id << "\n";
    double lat, lon;
    char buf[64];
    while (f >> lat >> lon) {
        double x, y, z;
        XPLMWorldToLocal(lat, lon, 0.0, &x, &y, &z);
        XPLMProbeInfo_t info;
        info.structSize = sizeof(info);
        if (XPLMProbeTerrainXYZ(g_probe, (float)x, (float)y, (float)z, &info) == xplm_ProbeHitTerrain) {
            double la, lo, alt;
            XPLMLocalToWorld(info.locationX, info.locationY, info.locationZ, &la, &lo, &alt);
            std::snprintf(buf, sizeof(buf), "%.2f\n", alt);
            out << buf;
        } else {
            out << "nan\n";
        }
    }
    std::string tmp = g_dir + "probe_ack.tmp", fin = g_dir + "probe_ack.txt";
    {
        std::ofstream o(tmp.c_str());
        o << out.str();
    }
    std::rename(tmp.c_str(), fin.c_str());
}

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
    handle_probe();
    std::string req = slurp(g_dir + "request.txt");
    size_t tab = req.find('\t');
    if (tab == std::string::npos) return 0.5f;
    std::string id = req.substr(0, tab);
    std::string path = req.substr(tab + 1);
    while (!path.empty() && (path.back() == '\n' || path.back() == '\r')) path.pop_back();
    int slot = 1;
    size_t tab2 = path.find('\t');
    if (tab2 != std::string::npos) {
        slot = std::atoi(path.c_str() + tab2 + 1);
        path = path.substr(0, tab2);
    }
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
    int total = 0, active = 0;
    XPLMPluginID owner;
    XPLMCountAircraft(&total, &active, &owner);
    if (slot < 1 || slot >= active) {   // active = user plane + AI planes in the flight
        write_ack(id, "fail", "AI slot " + std::to_string(slot) + " yok: Flight Configuration'da en az "
                  + std::to_string(slot) + " AI ucak sec");
        return 0.5f;
    }
    for (int i = 1; i < active; ++i) XPLMDisableAIForPlane(i);   // no AI plane flies into the picture
    XPLMSetAircraftModel(slot, path.c_str());
    XPLMDisableAIForPlane(slot);   // AI plane stays where Python puts it

    char file[256] = {0}, loaded[512] = {0};
    XPLMGetNthAircraftModel(slot, file, loaded);
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

PLUGIN_API void XPluginStop(void) {
    if (g_probe) { XPLMDestroyProbe(g_probe); g_probe = NULL; }
}
PLUGIN_API int XPluginEnable(void) {
    XPLMRegisterFlightLoopCallback(loop_cb, 0.5f, NULL);
    return 1;
}
PLUGIN_API void XPluginDisable(void) {
    XPLMUnregisterFlightLoopCallback(loop_cb, NULL);
    if (g_acquired) { XPLMReleasePlanes(); g_acquired = false; }
}
PLUGIN_API void XPluginReceiveMessage(XPLMPluginID, int, void *) {}
