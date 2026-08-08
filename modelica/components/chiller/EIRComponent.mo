model EIRComponent
  parameter Real QEva_flow_nominal(unit="W") = 4400000.0 annotation(Evaluate=false);
  parameter Real COP_nominal(unit="1") = 6.0 annotation(Evaluate=false);
  parameter Real mEva_flow_nominal(unit="kg/s") = 175.0 annotation(Evaluate=false);
  parameter Real mCon_flow_nominal(unit="kg/s") = 195.0 annotation(Evaluate=false);
  parameter Real PLRMin(unit="1") = 0.1 annotation(Evaluate=false);
  parameter Real PLRMax(unit="1") = 1.2 annotation(Evaluate=false);
  parameter Real yMin(unit="1") = 0.05 annotation(Evaluate=false);
  parameter Real etaMotor(unit="1") = 1.0 annotation(Evaluate=false);
  parameter Real cpWat(unit="J/(kg.K)") = 4180.0 annotation(Evaluate=false);
  parameter Real capFunT[6] = {0.8085529, 0.03412026, -0.0004184591, 0.004111323, -0.0001764627, 0.0004184748} annotation(Evaluate=false);
  parameter Real EIRFunT[6] = {0.6329473, 0.005521802, 0.0001487172, 0.002433576, 0.0007100129, -0.001108627} annotation(Evaluate=false);
  parameter Real EIRFunPLR[3] = {0.24019, 0.5090345, 0.2500532} annotation(Evaluate=false);

  Modelica.Blocks.Interfaces.RealInput TEvaEnt_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput TEvaLvgSet_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput TConEnt_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput mEva_flow_kg_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealInput mCon_flow_kg_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealInput y(unit="1");

  Modelica.Blocks.Interfaces.RealOutput TEvaLvg_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput TConLvg_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput QEva_flow_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput P_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput COP(unit="1");
  Modelica.Blocks.Interfaces.RealOutput PLR(unit="1");

protected
  Real y_s(unit="1");
  Real capFT(unit="1");
  Real eirFT(unit="1");
  Real eirFPLR(unit="1");
  Real availableCooling(unit="W");
  Real requestedCooling(unit="W");
  Real QCon_flow_W(unit="W");

equation
  y_s = min(max(y, 0.0), 1.0);
  capFT = max(0.05, capFunT[1] + capFunT[2] * TEvaLvgSet_C + capFunT[3] * TEvaLvgSet_C^2
    + capFunT[4] * TConEnt_C + capFunT[5] * TConEnt_C^2 + capFunT[6] * TEvaLvgSet_C * TConEnt_C);
  eirFT = max(0.05, EIRFunT[1] + EIRFunT[2] * TEvaLvgSet_C + EIRFunT[3] * TEvaLvgSet_C^2
    + EIRFunT[4] * TConEnt_C + EIRFunT[5] * TConEnt_C^2 + EIRFunT[6] * TEvaLvgSet_C * TConEnt_C);
  availableCooling = abs(QEva_flow_nominal) * capFT;
  requestedCooling = max(0.0, mEva_flow_kg_s) * cpWat * max(0.0, TEvaEnt_C - TEvaLvgSet_C);
  PLR = if y_s <= yMin then 0.0 else min(max(requestedCooling / max(1.0, availableCooling), PLRMin), PLRMax);
  eirFPLR = max(0.05, EIRFunPLR[1] + EIRFunPLR[2] * PLR + EIRFunPLR[3] * PLR^2);

  QEva_flow_W = availableCooling * PLR;
  P_W = if PLR <= 0.0 then 0.0 else availableCooling / max(0.1, COP_nominal) * eirFT * eirFPLR / max(0.1, etaMotor);
  COP = QEva_flow_W / max(1.0, P_W);
  TEvaLvg_C = TEvaEnt_C - QEva_flow_W / max(1.0, max(0.0, mEva_flow_kg_s) * cpWat);
  QCon_flow_W = QEva_flow_W + P_W;
  TConLvg_C = TConEnt_C + QCon_flow_W / max(1.0, max(0.0, mCon_flow_kg_s) * cpWat);

annotation(Documentation(info="<html><p>Signal-level ElectricEIR chiller component for generated system models. Calibrated curve coefficients are injected as parameters; operating conditions are explicit inputs.</p></html>"));
end EIRComponent;
