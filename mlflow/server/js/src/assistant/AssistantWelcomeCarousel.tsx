/**
 * Assistant Welcome Carousel component.
 * Displays a carousel of feature images with dot navigation.
 */

import { useEffect, useState } from 'react';
import { ChevronLeftIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';

import assistantDebugImg from '../common/static/assistant-debug.svg';
import assistantEvalImg from '../common/static/assistant-evaluation.svg';
import assistantTrendsImg from '../common/static/assistant-trends.svg';

interface CarouselSlide {
  image: string;
  title: string;
  description: string;
}

const slides: CarouselSlide[] = [
  {
    image: assistantDebugImg,
    title: 'Debug Issues',
    description: 'Ask questions about errors, identify root causes, and get actionable fixes for failed traces.',
  },
  {
    image: assistantEvalImg,
    title: 'Set Up Evaluations',
    description: 'Configure evaluation criteria, run assessments on your agents, and track quality metrics.',
  },
  {
    image: assistantTrendsImg,
    title: 'Analyze Trends',
    description: 'Explore metrics trends, uncover optimization opportunities, and get insights on your experiments.',
  },
];

const TRANSITION_DURATION_MS = 400;

export const AssistantWelcomeCarousel = () => {
  const { theme } = useDesignSystemTheme();
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isTransitioning, setIsTransitioning] = useState(false);

  const extendedSlides = [...slides, slides[0]];
  const displaySlideIndex = currentIndex % slides.length;

  // Reset transitioning state after animation completes
  useEffect(() => {
    if (!isTransitioning) return;

    const timeout = setTimeout(() => {
      // If we're at the clone (index 3), reset to first slide instantly
      if (currentIndex === slides.length) {
        setIsTransitioning(false);
        setCurrentIndex(0);
      } else {
        setIsTransitioning(false);
      }
    }, TRANSITION_DURATION_MS);
    return () => clearTimeout(timeout);
  }, [isTransitioning, currentIndex]);

  const handleNextSlide = () => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    setCurrentIndex((prev) => prev + 1);
  };

  const handlePrevSlide = () => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    setCurrentIndex((prev) => (prev - 1 + slides.length) % slides.length);
  };

  const handleDotClick = (index: number) => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    setCurrentIndex(index);
  };

  const currentSlideData = slides[displaySlideIndex];

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: theme.spacing.lg,
        width: '100%',
      }}
    >
      {/* Welcome header */}
      <div
        css={{
          textAlign: 'center',
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.sm,
          padding: `0 ${theme.spacing.lg}px`,
        }}
      >
        <Typography.Title level={3} css={{ margin: 0 }}>
          Welcome to MLflow Assistant
        </Typography.Title>
      </div>

      {/* Carousel container */}
      <div
        css={{
          position: 'relative',
          width: '100%',
          maxWidth: 480,
        }}
      >
        {/* Image area with arrow buttons */}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
          }}
        >
          {/* Left arrow */}
          <button
            onClick={handlePrevSlide}
            aria-label="Previous slide"
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: theme.spacing.lg * 1.5,
              height: theme.spacing.lg * 1.5,
              borderRadius: '50%',
              border: 'none',
              backgroundColor: 'transparent',
              color: theme.colors.textSecondary,
              cursor: 'pointer',
              flexShrink: 0,
              transition: 'all 0.2s ease',
              '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.textPrimary,
              },
            }}
          >
            <ChevronLeftIcon />
          </button>

          {/* Slides */}
          <div
            css={{
              flex: 1,
              overflow: 'hidden',
            }}
          >
            <div
              css={{
                display: 'flex',
                transition: isTransitioning ? `transform ${TRANSITION_DURATION_MS}ms ease-in-out` : 'none',
                transform: `translateX(-${currentIndex * 100}%)`,
              }}
            >
              {extendedSlides.map((slide, index) => (
                <div
                  key={`${slide.title}-${index}`}
                  css={{
                    flexShrink: 0,
                    width: '100%',
                  }}
                >
                  <img
                    src={slide.image}
                    alt={slide.title}
                    css={{
                      width: '100%',
                      height: 'auto',
                    }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Right arrow */}
          <button
            onClick={handleNextSlide}
            aria-label="Next slide"
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: theme.spacing.lg * 1.5,
              height: theme.spacing.lg * 1.5,
              borderRadius: '50%',
              border: 'none',
              backgroundColor: 'transparent',
              color: theme.colors.textSecondary,
              cursor: 'pointer',
              flexShrink: 0,
              transition: 'all 0.2s ease',
              '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
                color: theme.colors.textPrimary,
              },
            }}
          >
            <ChevronRightIcon />
          </button>
        </div>

        {/* Slide title and description with numbering */}
        <div
          css={{
            textAlign: 'center',
            marginTop: theme.spacing.md,
            padding: `0 32px`,
          }}
        >
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.xs }}>
            {displaySlideIndex + 1}. {currentSlideData.title}
          </Typography.Text>
          <Typography.Text color="secondary">{currentSlideData.description}</Typography.Text>
        </div>
      </div>

      {/* Dot indicators */}
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.sm,
          justifyContent: 'center',
        }}
      >
        {slides.map((slide, index) => (
          <button
            key={slide.title}
            onClick={() => handleDotClick(index)}
            aria-label={`Go to slide ${index + 1}: ${slide.title}`}
            css={{
              width: theme.spacing.sm,
              height: theme.spacing.sm,
              borderRadius: '50%',
              border: 'none',
              backgroundColor: theme.colors.textSecondary,
              opacity: index === displaySlideIndex ? 0.7 : 0.3,
              padding: 0,
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              '&:hover': {
                opacity: 0.7,
              },
            }}
          />
        ))}
      </div>
    </div>
  );
};
