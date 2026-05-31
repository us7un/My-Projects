/* * PacWarTk.c 
 * Robust Version: Safe String Formatting & Tcl 9 Compatibility
 */

#include "tk.h"
#include "PacWar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern Tcl_Interp *theInterp;
static char tclCmd[1024]; /* Large buffer to be safe */

enum simStatus {
  STEPPING,
  RUNNING,
  STOPPED,
  RESET,
  NUM_STAT
};
static char simStatusText[NUM_STAT][10] = {"step", "run", "stop", "reset"};

static int status = RESET;
static int duel = 1;

World w[2];
int clockCount = 0;
int count[2];
int order = 0;
PacGene g[2];
PacGenePtr gp[2] = {&g[0], &g[1]};

/* ******************************************************************
 * Execute Tcl command safely
 * *******************************************************************/
char *DoTclCmd(char *cmdStr)
{
  int rc;
  /* Use Tcl_EvalEx with exact length to avoid reading garbage */
  if ((rc = Tcl_EvalEx(theInterp, cmdStr, (int)strlen(cmdStr), 0)) != 0)
  {
    const char *result = Tcl_GetString(Tcl_GetObjResult(theInterp));
    printf("For tcl command '%s',\nError %d: %s\n", cmdStr, rc, result);
    return (char *)result;
  }
  return (0);
}

void DrawCell(int x, int y, Cell c)
{
  /* SAFE FORMATTING: Ensure null termination and no overflow */
  snprintf(tclCmd, sizeof(tclCmd), "DrawCell %d %d %d %d %d", x, y, c.kind, c.dir, c.age);
  DoTclCmd(tclCmd);
}

static void (*drawFcn)(int x, int y, Cell c) = &DrawCell;

/* *********************************************************************
 * Helper: Safely retrieve a gene string from the 'spec' array
 * *********************************************************************/
const char *GetGeneString(Tcl_Interp *interp, const char *keyVarName) {
    const char *geneName;
    const char *geneValue;

    geneName = Tcl_GetVar2(interp, keyVarName, NULL, TCL_GLOBAL_ONLY);
    if (geneName == NULL) {
        printf("Error: C could not find variable '%s'\n", keyVarName);
        return NULL;
    }

    geneValue = Tcl_GetVar2(interp, "spec", geneName, TCL_GLOBAL_ONLY);
    if (geneValue == NULL) {
        printf("Error: C could not find entry spec(%s)\n", geneName);
        return NULL;
    }
    return geneValue;
}

int RunSimTclCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *argv[])
{
  int newStatus;
  const char *kind, *show, *g1, *g2;

  if (argc != 2)
  {
    Tcl_SetObjResult(interp, Tcl_NewStringObj("wrong # args", -1));
    return TCL_ERROR;
  }

  if (strcmp(argv[1], simStatusText[RUNNING]) == 0)
    newStatus = RUNNING;
  else if (strcmp(argv[1], simStatusText[STEPPING]) == 0)
    newStatus = STEPPING;
  else
  {
    Tcl_SetObjResult(interp, Tcl_NewStringObj("Status must be step or run", -1));
    return TCL_ERROR;
  }

  theInterp = interp;

  if (status == RESET)
  {
    clockCount = 0;
    
    kind = Tcl_GetVar2(interp, "kind", NULL, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    if (kind == NULL) return TCL_ERROR;

    show = Tcl_GetVar2(interp, "showRounds", NULL, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    if (show == NULL) return TCL_ERROR;

    if (*show == '1')
      drawFcn = &DrawCell;
    else
      drawFcn = NULL;
    
    /* Get Genes Safely */
    g1 = GetGeneString(interp, "spec1name");
    if (g1 == NULL) {
      Tcl_SetObjResult(interp, Tcl_NewStringObj("Could not retrieve gene 1 string", -1));
      return TCL_ERROR;
    }
    if (SetGeneFromString((char*)g1, g) == 0) {
      Tcl_SetObjResult(interp, Tcl_NewStringObj("Gene-string 1 format was bad", -1));
      return TCL_ERROR;
    }
    count[0] = 1;
    order = 0;
    
    if (strcmp(kind, "duel") == 0)
    {
      g2 = GetGeneString(interp, "spec2name");
      if (g2 == NULL) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Could not retrieve gene 2 string", -1));
        return TCL_ERROR;
      }
      if (SetGeneFromString((char*)g2, g + 1) == 0) {
        Tcl_SetObjResult(interp, Tcl_NewStringObj("Gene-string 2 format was bad", -1));
        return TCL_ERROR;
      }
      count[1] = 1;
      duel = 1;
      PrepDuel(&(w[0]), &(w[1]), &DrawCell);
    }
    else if (strcmp(kind, "test") == 0)
    {
      count[1] = 0;
      duel = 0;
      PrepTest(&(w[0]), &(w[1]), &DrawCell);
    }
    else
    {
      Tcl_SetObjResult(interp, Tcl_NewStringObj("kind has invalid value", -1));
      return TCL_ERROR;
    }
    
    snprintf(tclCmd, sizeof(tclCmd), "%d", clockCount);
    Tcl_SetVar2(interp, "display_clockCount", NULL, tclCmd, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    
    snprintf(tclCmd, sizeof(tclCmd), "%d", count[0]);
    Tcl_SetVar2(interp, "spec1Cnt", NULL, tclCmd, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    
    if (duel)
    {
      snprintf(tclCmd, sizeof(tclCmd), "%d", count[1]);
      Tcl_SetVar2(interp, "spec2Cnt", NULL, tclCmd, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    }
    DoTclCmd("update");
    
    if (newStatus == STEPPING)
      status = STOPPED;
    else
      status = RUNNING;
  }
  else
    status = RUNNING;

  while (status == RUNNING)
  {
    ComputeNewWorld(&(w[order]), &(w[1 - order]), gp, count, drawFcn);
    order = 1 - order;
    clockCount++;
    
    snprintf(tclCmd, sizeof(tclCmd), "%d", clockCount);
    Tcl_SetVar2(interp, "display_clockCount", NULL, tclCmd, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    
    snprintf(tclCmd, sizeof(tclCmd), "%d", count[0]);
    Tcl_SetVar2(interp, "spec1Cnt", NULL, tclCmd, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    
    if (duel)
    {
      snprintf(tclCmd, sizeof(tclCmd), "%d", count[1]);
      Tcl_SetVar2(interp, "spec2Cnt", NULL, tclCmd, TCL_GLOBAL_ONLY | TCL_LEAVE_ERR_MSG);
    }
    DoTclCmd("update");
    if (status == RUNNING && (newStatus == STEPPING || clockCount == 500 ||
                              count[0] == 0 || (duel && count[1] == 0)))
      status = STOPPED;
  }
  if (status == STOPPED)
  {
    DoTclCmd("StopSimulation");
    if (drawFcn == NULL)
    {
      int x, y;
      for (x = 1; x < MaxX - 1; x++)
        for (y = 1; y < MaxY - 1; y++)
          DrawCell(x, y, w[order][x][y]);
    }
  }
  return TCL_OK;
}

int SetStatusTclCmd(ClientData clientData, Tcl_Interp *interp, int argc, const char *argv[])
{
  int i;

  if (argc != 2)
  {
    Tcl_SetObjResult(interp, Tcl_NewStringObj("Wrong # args", -1));
    return TCL_ERROR;
  }
  for (i = 1; i < NUM_STAT; i++)
    if (strcmp(argv[1], simStatusText[i]) == 0)
    {
      status = i;
      return TCL_OK;
    }
  Tcl_SetObjResult(interp, Tcl_NewStringObj("arg 1 must be either stop, run, or reset", -1));
  return TCL_ERROR;
}
