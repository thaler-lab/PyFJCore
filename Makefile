# If you are using this Makefile standalone and fastjet-config is not
# in your path, edit this line to specify the full path
CXX = g++
CXXFLAGS += -O3 -Wall -std=c++14 -fPIC -DPIC -DNDEBUG -DSWIG

#------------------------------------------------------------------------
# things that are specific to this contrib
NAME = PyFJCore
SRCDIR = pyfjcore
SRCS = fjcore.cc
#------------------------------------------------------------------------

SRCFILES := $(addprefix $(SRCDIR)/, $(SRCS))
OBJS := $(SRCFILES:.cc=.o)

ifeq "$(shell uname)" "Darwin"
    dynlibopt = -dynamiclib
    dynlibext = dylib
    #LDFLAGS += -install_name @rpath/lib$(NAME).dylib
else 
    dynlibopt = -shared
    dynlibext = so
endif

.PHONY: all shared clean

# http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/#combine
DEPDIR = $(SRCDIR)/.deps
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.d
COMPILE.cxx = $(CXX) $(DEPFLAGS) $(CXXFLAGS) -c

%.o : %.cc
$(SRCDIR)/%.o : $(SRCDIR)/%.cc $(DEPDIR)/%.d | $(DEPDIR)
	$(COMPILE.cxx) $(OUTPUT_OPTION) $<

# compilation of the code (default target)
all: shared
shared: lib$(NAME).$(dynlibext)

lib$(NAME).$(dynlibext): $(OBJS)
	$(CXX) $(OBJS) $(dynlibopt) $(CXXFLAGS) $(LDFLAGS) -g0 -o lib$(NAME).$(dynlibext)

# cleaning the directory
clean:
	rm -fv $(SRCDIR)/*.o *~ *.o *.a *.so *.dylib
	rm -rfv $(SRCDIR)/.deps *.dylib*

$(DEPDIR): ; @mkdir -p $@

DEPFILES := $(SRCS:%.cc=$(DEPDIR)/%.d)

$(DEPFILES):

include $(wildcard $(DEPFILES))
