#include <itkImageRegionConstIterator.h>
#include <itkAddImageFilter.h>
#include <itkSquareImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include "rtkTestConfiguration.h"
#include "rtkSheppLoganPhantomFilter.h"
#include "rtkDrawSheppLoganFilter.h"
#include "rtkConstantImageSource.h"
#include "itkShotNoiseImageFilter.h"
#include "rtkTest.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkFDKVarianceReconstructionFilter.h"

/**
 * \file rtkrampfiltertest.cxx
 *
 * \brief Functional test for the ramp filter of the FDK reconstruction.
 *
 * This test generates the projections of a simulated Shepp-Logan phantom in
 * different reconstruction scenarios (noise, truncation).
 * CT images are reconstructed from each set of projection images using the
 * FDK algorithm with different configuration of the ramp filter in order to
 * reduce the possible artifacts. The generated results are compared to the
 * expected results (analytical calculation).
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
  constexpr double hann = 1.0;
#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 3;
  constexpr unsigned int NumberOfSamples = 2;
#else
  constexpr unsigned int NumberOfProjectionImages = 180;
  constexpr unsigned int NumberOfSamples = 2000;
#endif

  // Constant image sources
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;
  ConstantImageSourceType::PointType   origin{ itk::MakePoint(-127.5, 0., -127.5) };
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer tomographySource = ConstantImageSourceType::New();
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, 2);
  spacing = itk::MakeVector(254., 254., 254.);
#else
  size = itk::MakeSize(256, 1, 256);
  spacing = itk::MakeVector(1., 1., 1.);
#endif
  tomographySource->SetOrigin(origin);
  tomographySource->SetSpacing(spacing);
  tomographySource->SetSize(size);
  tomographySource->SetConstant(0.);

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin = itk::MakePoint(-127.5, -1., -127.5);
#if FAST_TESTS_NO_CHECKS
  size = itk::MakeSize(2, 2, NumberOfProjectionImages);
  spacing = itk::MakeVector(508., 508., 508.);
#else
  size = itk::MakeSize(256, 3, NumberOfProjectionImages);
  spacing = itk::MakeVector(1., 1., 1.);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 0., noProj * 360. / NumberOfProjectionImages);

  // Shepp Logan projections filter
  using SLPType = rtk::SheppLoganPhantomFilter<OutputImageType, OutputImageType>;
  SLPType::Pointer slp = SLPType::New();
  slp->SetInput(projectionsSource->GetOutput());
  slp->SetGeometry(geometry);
  slp->InPlaceOff();

  std::cout << "\n\n****** Test : Create a set of noisy projections ******" << std::endl;

  // Add noise
  using NIFType = itk::ShotNoiseImageFilter<OutputImageType>;
  NIFType::Pointer noisy = NIFType::New();
  noisy->SetInput(slp->GetOutput());

  using AddType = itk::AddImageFilter<OutputImageType, OutputImageType>;
  AddType::Pointer add = AddType::New();
  add->InPlaceOff();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  OutputImageType::Pointer currentSum = tomographySource->GetOutput();
  currentSum->DisconnectPipeline();

  // TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionsSource->Update());
  // currentSum = projectionsSource->GetOutput();
  // currentSum->DisconnectPipeline();
  // slp->SetInput(projectionsSource->GetOutput());

  using SquareType = itk::SquareImageFilter<OutputImageType, OutputImageType>;
  SquareType::Pointer square = SquareType::New();
  AddType::Pointer    addSquare = AddType::New();
  addSquare->SetInput(1, square->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  OutputImageType::Pointer currentSumOfSquares = tomographySource->GetOutput();
  currentSumOfSquares->DisconnectPipeline();

  // TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionsSource->Update());
  // currentSumOfSquares = projectionsSource->GetOutput();
  // currentSumOfSquares->DisconnectPipeline();
  // slp->SetInput(projectionsSource->GetOutput());

  // FDK reconstruction
  using FDKType = rtk::FDKConeBeamReconstructionFilter<OutputImageType>;
  FDKType::Pointer feldkamp = FDKType::New();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(tomographySource->Update());
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, noisy->GetOutput());
  feldkamp->SetGeometry(geometry);
  feldkamp->GetRampFilter()->SetHannCutFrequency(hann);
  add->SetInput(1, feldkamp->GetOutput());
  square->SetInput(0, feldkamp->GetOutput());
  // add->SetInput(1, noisy->GetOutput());
  // square->SetInput(0, noisy->GetOutput());

  for (unsigned int i = 0; i < NumberOfSamples; ++i)
  {
    // New realization
    noisy->SetSeed(i);

    // Update sum
    add->SetInput(0, currentSum);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(add->Update());
    currentSum = add->GetOutput();
    currentSum->DisconnectPipeline();

    // Update sum of squared values
    addSquare->SetInput(0, currentSumOfSquares);
    addSquare->SetInput(1, square->GetOutput());
    TRY_AND_EXIT_ON_ITK_EXCEPTION(addSquare->Update());
    currentSumOfSquares = addSquare->GetOutput();
    currentSumOfSquares->DisconnectPipeline();
  }

  using MultiplyType = itk::MultiplyImageFilter<OutputImageType, OutputImageType>;
  MultiplyType::Pointer multiply = MultiplyType::New();
  multiply->SetInput(0, currentSumOfSquares);
  multiply->SetConstant((float)(1. / NumberOfSamples));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(multiply->Update());
  currentSumOfSquares = multiply->GetOutput();
  currentSumOfSquares->DisconnectPipeline();
  addSquare->SetInput(0, currentSumOfSquares);

  currentSum->DisconnectPipeline();
  square->SetInput(currentSum);
  multiply->SetInput(0, square->GetOutput());
  multiply->SetConstant((float)(-1. / (NumberOfSamples * NumberOfSamples)));
  addSquare->SetInput(1, multiply->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(addSquare->Update());

  itk::WriteImage(addSquare->GetOutput(), "ref.mha");

  using VarianceType = rtk::FDKVarianceReconstructionFilter<OutputImageType, OutputImageType>;
  VarianceType::Pointer variance = VarianceType::New();
  variance->SetGeometry(geometry);
  variance->SetInput(0, tomographySource->GetOutput());
  variance->SetInput(1, slp->GetOutput());
  variance->GetVarianceRampFilter()->SetHannCutFrequency(hann);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(variance->Update());
  itk::WriteImage(variance->GetOutput(), "var.mha");

  // CheckImageQuality<OutputImageType>(feldkamp->GetOutput(), dsl->GetOutput(), 0.72, 22.4, 2.);
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
