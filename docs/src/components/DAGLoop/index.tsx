import React, { useState, useRef } from 'react';
import { LucideIcon } from 'lucide-react';
import styles from './styles.module.css';

interface DAGStep {
  title: string;
  description: string;
  detailedDescription?: string;
  icon?: LucideIcon;
  highlight?: boolean;
  isFocus?: boolean;
}

interface DAGLoopProps {
  steps: DAGStep[];
  title?: string;
  loopBackIcon?: LucideIcon;
  loopBackText?: string;
  loopBackDescription?: string;
  circleSize?: number;
}

const DAGLoop: React.FC<DAGLoopProps> = ({
  steps,
  title,
  loopBackIcon: LoopIcon,
  loopBackText,
  loopBackDescription,
  circleSize = 400,
}) => {
  const [hoveredStep, setHoveredStep] = useState<number | null>(null);
  const [hoveredCenter, setHoveredCenter] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [isMobile, setIsMobile] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Detect mobile viewport
  React.useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Dynamic scaling based on number of elements
  const getOptimalRadius = () => {
    const baseRadius = circleSize / 2;
    const minRadius = isMobile ? 100 : 140;
    const maxRadius = isMobile ? 130 : 220;

    // Adjust radius based on number of steps
    // More steps = larger radius for better spacing
    const scaleFactor = Math.min(1.2, 0.8 + steps.length * 0.05);
    const calculatedRadius = (baseRadius - (isMobile ? 50 : 80)) * scaleFactor;

    return Math.max(minRadius, Math.min(maxRadius, calculatedRadius));
  };

  // Adjust size for mobile and dynamic scaling
  const actualCircleSize = isMobile ? 280 : circleSize;
  const radius = getOptimalRadius();
  const centerX = actualCircleSize / 2;
  const centerY = actualCircleSize / 2;

  const calculatePosition = (index: number) => {
    const angle = (index * 2 * Math.PI) / steps.length - Math.PI / 2;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    return { x, y };
  };

  const calculateArrowPath = (fromIndex: number, toIndex: number) => {
    // Get positions of both nodes
    const fromPos = calculatePosition(fromIndex);
    const toPos = calculatePosition(toIndex);

    // Calculate vector from center to center
    const dx = toPos.x - fromPos.x;
    const dy = toPos.y - fromPos.y;

    // Calculate the midpoint between the two nodes
    const midX = (fromPos.x + toPos.x) / 2;
    const midY = (fromPos.y + toPos.y) / 2;

    // Normalize the direction vector
    const length = Math.sqrt(dx * dx + dy * dy);
    const dirX = dx / length;
    const dirY = dy / length;

    // Short arrow positioned at midpoint
    const arrowLength = 20; // Total length of arrow line

    // Start point of arrow (back from midpoint along the direction vector)
    const startX = midX - (arrowLength / 2) * dirX;
    const startY = midY - (arrowLength / 2) * dirY;

    // End point of arrow (forward from midpoint along the direction vector)
    const endX = midX + (arrowLength / 2) * dirX;
    const endY = midY + (arrowLength / 2) * dirY;

    // Simple straight line
    return `M ${startX} ${startY} L ${endX} ${endY}`;
  };

  const handleMouseEnter = (index: number, event: React.MouseEvent) => {
    setHoveredStep(index);
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const position = calculatePosition(index);
      setTooltipPosition({
        x: position.x,
        y: position.y,
      });
    }
  };

  const handleMouseLeave = () => {
    setHoveredStep(null);
  };

  // Render linear layout for mobile
  if (isMobile) {
    return (
      <div className={styles.loopContainer}>
        {title && <h3 className={styles.loopTitle}>{title}</h3>}

        <div className={styles.mobileLinearContent}>
          {steps.map((step, index) => (
            <div key={index} className={styles.mobileStepItem}>
              <div className={styles.mobileStepIndicator}>
                <div className={`${styles.mobileStepNumber} ${step.isFocus ? styles.mobileFocusNode : ''}`}>
                  {step.icon ? <step.icon size={20} /> : <span>{index + 1}</span>}
                </div>
                {index < steps.length - 1 && <div className={styles.mobileStepConnector} />}
              </div>
              <div className={styles.mobileStepContent}>
                <h4 className={styles.mobileStepTitle}>{step.title}</h4>
                <p className={styles.mobileStepDescription}>{step.detailedDescription || step.description}</p>
              </div>
            </div>
          ))}

          {LoopIcon && loopBackDescription && (
            <div className={styles.mobileLoopBack}>
              <div className={styles.mobileLoopIcon}>
                <LoopIcon size={24} />
              </div>
              <div className={styles.mobileLoopContent}>
                <h4 className={styles.mobileLoopTitle}>{loopBackText || 'Iterate'}</h4>
                <p className={styles.mobileLoopDescription}>{loopBackDescription}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Desktop circular layout
  return (
    <div className={styles.loopContainer}>
      {title && <h3 className={styles.loopTitle}>{title}</h3>}

      <div className={styles.loopContent}>
        <div
          className={styles.circleContainer}
          ref={containerRef}
          style={{
            width: `${actualCircleSize}px`,
            height: `${actualCircleSize}px`,
          }}
        >
          <svg width={actualCircleSize} height={actualCircleSize} className={styles.svgCanvas}>
            {/* Draw arrows between steps */}
            {steps.map((_, index) => {
              const nextIndex = (index + 1) % steps.length;
              return (
                <g key={`arrow-${index}`}>
                  <defs>
                    <marker id={`arrowhead-${index}`} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                      <path
                        d="M 0 0 L 6 3 L 0 6 L 1.5 3 Z"
                        fill="currentColor"
                        opacity="1"
                        className={styles.arrowHead}
                      />
                    </marker>
                  </defs>
                  <path
                    d={calculateArrowPath(index, nextIndex)}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeDasharray="0"
                    opacity="0.9"
                    markerEnd={`url(#arrowhead-${index})`}
                    className={styles.arrowPath}
                  />
                </g>
              );
            })}

            {/* Draw loop indicator in center */}
            {LoopIcon && (
              <g
                className={styles.centerIcon}
                onMouseEnter={() => setHoveredCenter(true)}
                onMouseLeave={() => setHoveredCenter(false)}
                style={{ cursor: 'pointer' }}
              >
                <foreignObject x={centerX - 35} y={centerY - 35} width="70" height="70">
                  <div className={styles.loopIconWrapper}>
                    <LoopIcon size={32} />
                  </div>
                </foreignObject>
                {loopBackText && (
                  <text x={centerX} y={centerY + 50} textAnchor="middle" className={styles.loopText}>
                    {loopBackText}
                  </text>
                )}
              </g>
            )}
          </svg>

          {/* Render step nodes */}
          {steps.map((step, index) => {
            const position = calculatePosition(index);
            return (
              <div
                key={index}
                className={`${styles.stepNode} ${step.highlight ? styles.highlighted : ''} ${step.isFocus ? styles.focusNode : ''}`}
                style={{
                  left: `${position.x}px`,
                  top: `${position.y}px`,
                  transform: 'translate(-50%, -50%)',
                }}
                onMouseEnter={(e) => handleMouseEnter(index, e)}
                onMouseLeave={handleMouseLeave}
              >
                <div className={styles.stepNodeContent}>
                  {step.icon ? <step.icon size={24} /> : <span className={styles.stepNumber}>{index + 1}</span>}
                </div>
                <div className={styles.stepLabel}>{step.title}</div>
              </div>
            );
          })}

          {/* Tooltip for steps */}
          {hoveredStep !== null && (
            <div
              className={styles.tooltip}
              style={{
                left: `${tooltipPosition.x}px`,
                top: `${tooltipPosition.y}px`,
                transform: 'translate(-50%, -120%)',
              }}
            >
              <h4 className={styles.tooltipTitle}>{steps[hoveredStep].title}</h4>
              <p className={styles.tooltipDescription}>
                {steps[hoveredStep].detailedDescription || steps[hoveredStep].description}
              </p>
              <div className={styles.tooltipArrow} />
            </div>
          )}

          {/* Tooltip for center icon */}
          {hoveredCenter && loopBackDescription && (
            <div
              className={styles.centerTooltip}
              style={{
                left: `${centerX}px`,
                top: `${centerY}px`,
                transform: 'translate(-50%, -50%)',
              }}
            >
              <p className={styles.centerTooltipDescription}>{loopBackDescription}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DAGLoop;
