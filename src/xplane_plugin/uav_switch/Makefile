# Build:  make            (needs g++; uses the XPLM headers from the XPlaneConnect checkout)
# Install: make install   (copies to X-Plane 11/Resources/plugins/UavSwitch/64/lin.xpl)
SDK ?= $(HOME)/Belgeler/GitHub/XPlaneConnect/xpcPlugin/SDK/CHeaders/XPLM
XP  ?= $(HOME)/.local/share/Steam/steamapps/common/X-Plane 11

lin.xpl: uav_switch.cpp
	g++ -shared -fPIC -m64 -O2 -fno-stack-protector -fvisibility=hidden -std=c++11 \
	    -DXPLM200 -DXPLM210 -DLIN=1 -I$(SDK) -o $@ $<

install: lin.xpl
	mkdir -p "$(XP)/Resources/plugins/UavSwitch/64" "$(XP)/Output/uav_switch"
	cp lin.xpl "$(XP)/Resources/plugins/UavSwitch/64/lin.xpl"

clean:
	rm -f lin.xpl
