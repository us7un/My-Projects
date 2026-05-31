#include "tk.h"
#include <stdlib.h>

Tcl_Interp *theInterp = NULL;

extern int RunSimTclCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *argv[]);
extern int SetStatusTclCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *argv[]);

int main(int argc, char **argv) {
    Tk_Main(argc, argv, Tcl_AppInit);
    return 0;
}

int Tcl_AppInit(Tcl_Interp *interp) {
    if (Tcl_Init(interp) == TCL_ERROR) return TCL_ERROR;
    if (Tk_Init(interp) == TCL_ERROR) return TCL_ERROR;

    theInterp = interp;
    Tcl_CreateCommand(interp, "RunSim", RunSimTclCmd, (ClientData)NULL, (Tcl_CmdDeleteProc *)NULL);
    Tcl_CreateCommand(interp, "SetStatus", SetStatusTclCmd, (ClientData)NULL, (Tcl_CmdDeleteProc *)NULL);

    return TCL_OK;
}
