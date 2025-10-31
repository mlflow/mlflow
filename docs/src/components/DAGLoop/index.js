import React, { useState, useRef } from 'react';
import styles from './styles.module.css';
var DAGLoop = function (_a) {
    var steps = _a.steps, title = _a.title, LoopIcon = _a.loopBackIcon, loopBackText = _a.loopBackText, loopBackDescription = _a.loopBackDescription, _b = _a.circleSize, circleSize = _b === void 0 ? 400 : _b;
    var _c = useState(null), hoveredStep = _c[0], setHoveredStep = _c[1];
    var _d = useState(false), hoveredCenter = _d[0], setHoveredCenter = _d[1];
    var _e = useState({ x: 0, y: 0 }), tooltipPosition = _e[0], setTooltipPosition = _e[1];
    var _f = useState(false), isMobile = _f[0], setIsMobile = _f[1];
    var containerRef = useRef(null);
    // Detect mobile viewport
    React.useEffect(function () {
        var checkMobile = function () {
            setIsMobile(window.innerWidth <= 768);
        };
        checkMobile();
        window.addEventListener('resize', checkMobile);
        return function () { return window.removeEventListener('resize', checkMobile); };
    }, []);
    // Dynamic scaling based on number of elements
    var getOptimalRadius = function () {
        var baseRadius = circleSize / 2;
        var minRadius = isMobile ? 100 : 140;
        var maxRadius = isMobile ? 130 : 220;
        // Adjust radius based on number of steps
        // More steps = larger radius for better spacing
        var scaleFactor = Math.min(1.2, 0.8 + steps.length * 0.05);
        var calculatedRadius = (baseRadius - (isMobile ? 50 : 80)) * scaleFactor;
        return Math.max(minRadius, Math.min(maxRadius, calculatedRadius));
    };
    // Adjust size for mobile and dynamic scaling
    var actualCircleSize = isMobile ? 280 : circleSize;
    var radius = getOptimalRadius();
    var centerX = actualCircleSize / 2;
    var centerY = actualCircleSize / 2;
    var calculatePosition = function (index) {
        var angle = (index * 2 * Math.PI) / steps.length - Math.PI / 2;
        var x = centerX + radius * Math.cos(angle);
        var y = centerY + radius * Math.sin(angle);
        return { x: x, y: y };
    };
    var calculateArrowPath = function (fromIndex, toIndex) {
        // Get positions of both nodes
        var fromPos = calculatePosition(fromIndex);
        var toPos = calculatePosition(toIndex);
        // Calculate vector from center to center
        var dx = toPos.x - fromPos.x;
        var dy = toPos.y - fromPos.y;
        // Calculate the midpoint between the two nodes
        var midX = (fromPos.x + toPos.x) / 2;
        var midY = (fromPos.y + toPos.y) / 2;
        // Normalize the direction vector
        var length = Math.sqrt(dx * dx + dy * dy);
        var dirX = dx / length;
        var dirY = dy / length;
        // Short arrow positioned at midpoint
        var arrowLength = 20; // Total length of arrow line
        // Start point of arrow (back from midpoint along the direction vector)
        var startX = midX - (arrowLength / 2) * dirX;
        var startY = midY - (arrowLength / 2) * dirY;
        // End point of arrow (forward from midpoint along the direction vector)
        var endX = midX + (arrowLength / 2) * dirX;
        var endY = midY + (arrowLength / 2) * dirY;
        // Simple straight line
        return "M ".concat(startX, " ").concat(startY, " L ").concat(endX, " ").concat(endY);
    };
    var handleMouseEnter = function (index, event) {
        setHoveredStep(index);
        if (containerRef.current) {
            var rect = containerRef.current.getBoundingClientRect();
            var position = calculatePosition(index);
            setTooltipPosition({
                x: position.x,
                y: position.y,
            });
        }
    };
    var handleMouseLeave = function () {
        setHoveredStep(null);
    };
    // Render linear layout for mobile
    if (isMobile) {
        return (<div className={styles.loopContainer}>
        {title && <h3 className={styles.loopTitle}>{title}</h3>}

        <div className={styles.mobileLinearContent}>
          {steps.map(function (step, index) { return (<div key={index} className={styles.mobileStepItem}>
              <div className={styles.mobileStepIndicator}>
                <div className={"".concat(styles.mobileStepNumber, " ").concat(step.isFocus ? styles.mobileFocusNode : '')}>
                  {step.icon ? <step.icon size={20}/> : <span>{index + 1}</span>}
                </div>
                {index < steps.length - 1 && <div className={styles.mobileStepConnector}/>}
              </div>
              <div className={styles.mobileStepContent}>
                <h4 className={styles.mobileStepTitle}>{step.title}</h4>
                <p className={styles.mobileStepDescription}>{step.detailedDescription || step.description}</p>
              </div>
            </div>); })}

          {LoopIcon && loopBackDescription && (<div className={styles.mobileLoopBack}>
              <div className={styles.mobileLoopIcon}>
                <LoopIcon size={24}/>
              </div>
              <div className={styles.mobileLoopContent}>
                <h4 className={styles.mobileLoopTitle}>{loopBackText || 'Iterate'}</h4>
                <p className={styles.mobileLoopDescription}>{loopBackDescription}</p>
              </div>
            </div>)}
        </div>
      </div>);
    }
    // Desktop circular layout
    return (<div className={styles.loopContainer}>
      {title && <h3 className={styles.loopTitle}>{title}</h3>}

      <div className={styles.loopContent}>
        <div className={styles.circleContainer} ref={containerRef} style={{
            width: "".concat(actualCircleSize, "px"),
            height: "".concat(actualCircleSize, "px"),
        }}>
          <svg width={actualCircleSize} height={actualCircleSize} className={styles.svgCanvas}>
            {/* Draw arrows between steps */}
            {steps.map(function (_, index) {
            var nextIndex = (index + 1) % steps.length;
            return (<g key={"arrow-".concat(index)}>
                  <defs>
                    <marker id={"arrowhead-".concat(index)} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                      <path d="M 0 0 L 6 3 L 0 6 L 1.5 3 Z" fill="currentColor" opacity="1" className={styles.arrowHead}/>
                    </marker>
                  </defs>
                  <path d={calculateArrowPath(index, nextIndex)} fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="0" opacity="0.9" markerEnd={"url(#arrowhead-".concat(index, ")")} className={styles.arrowPath}/>
                </g>);
        })}

            {/* Draw loop indicator in center */}
            {LoopIcon && (<g className={styles.centerIcon} onMouseEnter={function () { return setHoveredCenter(true); }} onMouseLeave={function () { return setHoveredCenter(false); }} style={{ cursor: 'pointer' }}>
                <foreignObject x={centerX - 35} y={centerY - 35} width="70" height="70">
                  <div className={styles.loopIconWrapper}>
                    <LoopIcon size={32}/>
                  </div>
                </foreignObject>
                {loopBackText && (<text x={centerX} y={centerY + 50} textAnchor="middle" className={styles.loopText}>
                    {loopBackText}
                  </text>)}
              </g>)}
          </svg>

          {/* Render step nodes */}
          {steps.map(function (step, index) {
            var position = calculatePosition(index);
            return (<div key={index} className={"".concat(styles.stepNode, " ").concat(step.highlight ? styles.highlighted : '', " ").concat(step.isFocus ? styles.focusNode : '')} style={{
                    left: "".concat(position.x, "px"),
                    top: "".concat(position.y, "px"),
                    transform: 'translate(-50%, -50%)',
                }} onMouseEnter={function (e) { return handleMouseEnter(index, e); }} onMouseLeave={handleMouseLeave}>
                <div className={styles.stepNodeContent}>
                  {step.icon ? <step.icon size={24}/> : <span className={styles.stepNumber}>{index + 1}</span>}
                </div>
                <div className={styles.stepLabel}>{step.title}</div>
              </div>);
        })}

          {/* Tooltip for steps */}
          {hoveredStep !== null && (<div className={styles.tooltip} style={{
                left: "".concat(tooltipPosition.x, "px"),
                top: "".concat(tooltipPosition.y, "px"),
                transform: 'translate(-50%, -120%)',
            }}>
              <h4 className={styles.tooltipTitle}>{steps[hoveredStep].title}</h4>
              <p className={styles.tooltipDescription}>
                {steps[hoveredStep].detailedDescription || steps[hoveredStep].description}
              </p>
              <div className={styles.tooltipArrow}/>
            </div>)}

          {/* Tooltip for center icon */}
          {hoveredCenter && loopBackDescription && (<div className={styles.centerTooltip} style={{
                left: "".concat(centerX, "px"),
                top: "".concat(centerY, "px"),
                transform: 'translate(-50%, -50%)',
            }}>
              <p className={styles.centerTooltipDescription}>{loopBackDescription}</p>
            </div>)}
        </div>
      </div>
    </div>);
};
export default DAGLoop;
