model CarnotComponent
  parameter Real QEva_flow_nominal(unit="W") = 4400000.0 annotation(Evaluate=false);
  parameter Real etaCarnot_nominal(unit="1") = 0.45 annotation(Evaluate=false);
  parameter Real mEva_flow_nominal(unit="kg/s") = 175.0 annotation(Evaluate=false);
  parameter Real mCon_flow_nominal(unit="kg/s") = 195.0 annotation(Evaluate=false);
  parameter Real PLRMin(unit="1") = 0.05 annotation(Evaluate=false);
  parameter Real PLRMax(unit="1") = 1.2 annotation(Evaluate=false);
  parameter Real yMin(unit="1") = 0.05 annotation(Evaluate=false);
  parameter Real cpWat(unit="J/(kg.K)") = 4180.0 annotation(Evaluate=false);
  parameter Real a[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0} annotation(Evaluate=false);

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
  Real requestedCooling(unit="W");
  Real etaPL(unit="1");
  Real carnotCOP(unit="1");
  Real QCon_flow_W(unit="W");

equation
  y_s = min(max(y, 0.0), 1.0);
  requestedCooling = max(0.0, mEva_flow_kg_s) * cpWat * max(0.0, TEvaEnt_C - TEvaLvgSet_C);
  PLR = if y_s <= yMin then 0.0 else min(max(requestedCooling / max(1.0, abs(QEva_flow_nominal)), PLRMin), PLRMax);
  etaPL = max(0.05, a[1] + a[2] * PLR + a[3] * PLR^2 + a[4] * PLR^3 + a[5] * PLR^4 + a[6] * PLR^5);
  carnotCOP = max(0.1, etaCarnot_nominal * etaPL * (TEvaLvgSet_C + 273.15) / max(1.0, TConEnt_C - TEvaLvgSet_C));

  QEva_flow_W = abs(QEva_flow_nominal) * PLR;
  P_W = if PLR <= 0.0 then 0.0 else QEva_flow_W / carnotCOP;
  COP = QEva_flow_W / max(1.0, P_W);
  TEvaLvg_C = TEvaEnt_C - QEva_flow_W / max(1.0, max(0.0, mEva_flow_kg_s) * cpWat);
  QCon_flow_W = QEva_flow_W + P_W;
  TConLvg_C = TConEnt_C + QCon_flow_W / max(1.0, max(0.0, mCon_flow_kg_s) * cpWat);

annotation(Documentation(info="<html><p>Signal-level Carnot chiller component for generated system models. It exposes Carnot effectiveness and part-load curve parameters for calibrated assembly.</p></html>"));
end CarnotComponent;
