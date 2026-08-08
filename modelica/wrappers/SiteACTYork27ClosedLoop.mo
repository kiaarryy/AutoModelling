model SiteACTYork27ClosedLoop
  "Site A cooling tower, York 27-coefficient correlation, MBL-consistent closed loop"

  // ---------------------------------------------------------------------------
  // Corrects three defects of the shipped SiteACTYork27 wrapper.  See
  // docs/REVISION_ENERGY_01_AUDIT.md section B.
  //
  // 1. The range is the model's OWN range.  The shipped wrapper read TRan from
  //    the measured input table, where it is Tin - Tout_measured, so the
  //    predicted outlet was a function of the measured outlet.  MBL 12.1.0
  //    YorkCalc.mo line 20 defines TRan = T_a - T_b from the component's own
  //    outlet port and lets the solver close the resulting algebraic loop; this
  //    model does the same.
  //
  // 2. FRWat0 is applied.  MBL back-solves it in an initial equation from the
  //    design approach and normalises the water flow by
  //    mWat_flow_nominal = m_flow_nominal/FRWat0.  Omitting it put the liquid-
  //    to-gas ratio on a different axis from the one the default coefficients
  //    were derived on (Site A back-solves 0.785 to 1.375).
  //
  // 3. A free-convection branch exists.  MBL blends the forced-convection
  //    approach with a free-convection one around y = yMin; the shipped wrapper
  //    only clipped the fan speed.
  //
  // Measured Tout, TRan and TAppAct remain in the input table but feed only the
  // *_m scoring outputs.  Nothing measured downstream of the tower reaches the
  // prediction path.
  //
  // Deliberately free of any Buildings dependency: only MSL blocks are used, so
  // the FMU can be exported with OpenModelica as well as Dymola.
  // ---------------------------------------------------------------------------

  parameter String table_path = "CT_01_full_period_table.txt" annotation(Evaluate=false);
  parameter Real m_flow_nominal(unit="kg/s") = 100.0 annotation(Evaluate=false);
  parameter Real FRWat0 = 1.0
    "MBL water-flow-ratio normalisation, back-solved from the design point"
    annotation(Evaluate=false);
  parameter Real yMin(min=0, max=1) = 0.05 annotation(Evaluate=false);
  parameter Real fraFreCon(min=0, max=1) = 0.1
    "Fraction of nominal capacity available under free convection"
    annotation(Evaluate=false);
  parameter Real RlgMin(min=0) = 0.05 annotation(Evaluate=false);
  parameter Real RlgMax(min=0) = 8.0 annotation(Evaluate=false);
  parameter Real TAppMin_C = 0.0 annotation(Evaluate=false);
  parameter Real TAppMax_C = 40.0 annotation(Evaluate=false);
  parameter Real cpWat(unit="J/(kg.K)") = 4180.0 annotation(Evaluate=false);
  parameter Real PFan_nominal(unit="W") = 10000.0 annotation(Evaluate=false);
  parameter Real fanRelPow_r_P[5] = {0.0, 0.0025, 0.0437, 0.265, 1.0} annotation(Evaluate=false);

  // MBL Correlations/yorkCalc.mo c[1..27]; overwritten per tower by the runner
  parameter Real f[27] = {
    -0.359741205, -0.055053608, 0.0023850432,
    0.173926877, -0.0248473764, 0.00048430224,
    -0.005589849456, 0.0005770079712, -0.00001342427256,
    2.84765801111111, -0.121765149, 0.0014599242,
    1.680428651, -0.0166920786, -0.0007190532,
    -0.025485194448, 0.0000487491696, 0.00002719234152,
    -0.0653766255555556, -0.002278167, 0.0002500254,
    -0.0910565458, 0.00318176316, 0.000038621772,
    -0.0034285382352, 0.00000856589904, -0.000001516821552} annotation(Evaluate=false);

  // MBL Correlations/BoundsYorkCalc.mo -- reported, not enforced
  parameter Real TRanValidMin = 1.1 annotation(Evaluate=false);
  parameter Real TRanValidMax = 22.2 annotation(Evaluate=false);
  parameter Real FRWatValidMin = 0.75 annotation(Evaluate=false);
  parameter Real FRWatValidMax = 1.25 annotation(Evaluate=false);

  Modelica.Blocks.Sources.CombiTimeTable tab(
    tableOnFile=true,
    tableName="CT_data",
    fileName=table_path,
    columns=2:12,
    smoothness=Modelica.Blocks.Types.Smoothness.ConstantSegments,
    extrapolation=Modelica.Blocks.Types.Extrapolation.HoldLastPoint);

  Real Tin_C, Tout_meas_C, Twb_C, TRan_meas_C, TAppAct_C, mdot_cell_kgps, y_used;
  Real FRWat, FRAir, Rlg;
  Real TRan_s_C "Model-predicted range -- the closed loop";
  Real TApp_forced_C, TApp_free_C, TApp_raw_C;
  Real fanRelPow;

  Modelica.Blocks.Interfaces.RealOutput TOut_m(unit="K");
  Modelica.Blocks.Interfaces.RealOutput TOut_s(unit="K");
  Modelica.Blocks.Interfaces.RealOutput Q_m(unit="W");
  Modelica.Blocks.Interfaces.RealOutput Q_s(unit="W");
  Modelica.Blocks.Interfaces.RealOutput P_m(unit="W");
  Modelica.Blocks.Interfaces.RealOutput P_s(unit="W");
  Modelica.Blocks.Interfaces.RealOutput TApp_m(unit="K");
  Modelica.Blocks.Interfaces.RealOutput TApp_s(unit="K");
  Modelica.Blocks.Interfaces.RealOutput Rlg_s(unit="1");
  Modelica.Blocks.Interfaces.RealOutput inDomain(unit="1")
    "1 when the operating point is inside the York correlation's validity box";

protected
  function yorkApproach "27-term York approach-temperature correlation"
    input Real Twb "Wet-bulb temperature, degC";
    input Real TRan "Range temperature, K";
    input Real Rlg "Liquid-to-gas ratio";
    input Real c[27];
    output Real TApp "Approach temperature, K";
  algorithm
    TApp := c[1] + c[2]*Twb + c[3]*Twb^2
          + c[4]*TRan + c[5]*Twb*TRan + c[6]*Twb^2*TRan
          + c[7]*TRan^2 + c[8]*Twb*TRan^2 + c[9]*Twb^2*TRan^2
          + c[10]*Rlg + c[11]*Twb*Rlg + c[12]*Twb^2*Rlg
          + c[13]*TRan*Rlg + c[14]*Twb*TRan*Rlg + c[15]*Twb^2*TRan*Rlg
          + c[16]*TRan^2*Rlg + c[17]*Twb*TRan^2*Rlg + c[18]*Twb^2*TRan^2*Rlg
          + c[19]*Rlg^2 + c[20]*Twb*Rlg^2 + c[21]*Twb^2*Rlg^2
          + c[22]*TRan*Rlg^2 + c[23]*Twb*TRan*Rlg^2 + c[24]*Twb^2*TRan*Rlg^2
          + c[25]*TRan^2*Rlg^2 + c[26]*Twb*TRan^2*Rlg^2 + c[27]*Twb^2*TRan^2*Rlg^2;
  end yorkApproach;

equation
  Tin_C          = tab.y[1];
  Tout_meas_C    = tab.y[2];
  Twb_C          = tab.y[3];
  TRan_meas_C    = tab.y[4];
  TAppAct_C      = tab.y[5];
  mdot_cell_kgps = max(0.0, tab.y[6]);
  y_used         = tab.y[7];

  FRWat = mdot_cell_kgps / max(1e-6, m_flow_nominal) * FRWat0;
  FRAir = min(max(y_used, yMin), 1.0);
  Rlg   = min(max(FRWat / max(1e-4, FRAir), RlgMin), RlgMax);

  // closed loop: the range follows from the model's own outlet temperature
  TRan_s_C = Tin_C - (Twb_C + TApp_s);

  TApp_forced_C = yorkApproach(Twb_C, TRan_s_C, Rlg, f);
  // free convection: the tower still rejects some heat with the fan off
  TApp_free_C = (1.0 - fraFreCon) * max(0.0, Tin_C - Twb_C)
              + fraFreCon * yorkApproach(Twb_C, TRan_s_C,
                  min(max(FRWat, RlgMin), RlgMax), f);
  TApp_raw_C = Modelica.Fluid.Utilities.regStep(
                 y_used - yMin, TApp_forced_C, TApp_free_C, yMin/20.0);

  TApp_s = min(max(TApp_raw_C, TAppMin_C), TAppMax_C);

  TOut_s = Twb_C + TApp_s + 273.15;
  TOut_m = Tout_meas_C + 273.15;
  Q_s    = mdot_cell_kgps * cpWat * TRan_s_C;
  Q_m    = tab.y[10];
  P_s    = PFan_nominal * fanRelPow;
  P_m    = tab.y[11];
  TApp_m = TAppAct_C;
  Rlg_s  = Rlg;

  inDomain = if (TRan_s_C >= TRanValidMin and TRan_s_C <= TRanValidMax
                 and FRWat >= FRWatValidMin and FRWat <= FRWatValidMax)
             then 1.0 else 0.0;

  fanRelPow = if FRAir <= 0.1 then
      fanRelPow_r_P[1] + (fanRelPow_r_P[2] - fanRelPow_r_P[1]) * FRAir / 0.1
    elseif FRAir <= 0.3 then
      fanRelPow_r_P[2] + (fanRelPow_r_P[3] - fanRelPow_r_P[2]) * (FRAir - 0.1) / 0.2
    elseif FRAir <= 0.6 then
      fanRelPow_r_P[3] + (fanRelPow_r_P[4] - fanRelPow_r_P[3]) * (FRAir - 0.3) / 0.3
    else
      fanRelPow_r_P[4] + (fanRelPow_r_P[5] - fanRelPow_r_P[4]) * (FRAir - 0.6) / 0.4;

annotation(
  experiment(StartTime=0, StopTime=3600, Interval=300, Tolerance=1e-6),
  Documentation(info="<html>
<p>MBL-consistent replacement for <code>SiteACTYork27</code>. The 27 York
coefficients stay exposed as tunable FMI start values, but the range that drives
them is the model's own, so the exported FMU is a free-running simulation rather
than a one-step predictor. Regression oracle:
<code>autofmu.devices.york27_reference.closed_loop</code>.</p>
</html>"));
end SiteACTYork27ClosedLoop;
