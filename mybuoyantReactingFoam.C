/*---------------------------------------------------------------------------*\
  Application
    ammoniaDiffusionFoam

  Description
    Transient solver for turbulent flow and diffusion of ammonia in air.
    This solver is a simplified version of buoyantReactingFoam,
    removing combustion, reaction, and compressible flow components.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvModels.H"
#include "fvConstraints.H"

int main(int argc, char *argv[])
{
    // Initialize OpenFOAM environment
    Foam::argList args(argc, argv);
    Foam::Time runTime(Foam::Time::controlDictName, args);
    Foam::fvMesh mesh(Foam::IOobject("mesh", runTime.timeName(), runTime, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE));

    // Velocity and ammonia concentration fields
    Foam::volVectorField U(
        Foam::IOobject("U", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
        mesh
    );

    Foam::volScalarField NH3(
        Foam::IOobject("NH3", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
        mesh
    );

    // Surface flux field
    Foam::surfaceScalarField phi(
        Foam::IOobject("phi", runTime.timeName(), mesh, Foam::IOobject::MUST_READ, Foam::IOobject::AUTO_WRITE),
        mesh
    );

    // Time loop
    while (runTime.run())
    {
        runTime++; // Advance time step

        // Solve velocity equation (ignoring buoyancy and reaction terms)
        Foam::tmp<Foam::fvVectorMatrix> UEqn(
            Foam::fvm::ddt(U) + Foam::fvm::div(phi, U)
        );

        UEqn.ref().relax();
        Foam::solve(UEqn == -Foam::fvc::grad(NH3));

        // Solve ammonia diffusion equation
        Foam::solve(Foam::fvm::ddt(NH3) + Foam::fvm::div(phi, NH3) - Foam::fvm::laplacian(Foam::dimensionedScalar("D", Foam::dimViscosity, 1e-5), NH3));

        runTime.write(); // Write results
    }

    Info << "Simulation complete." << endl;

    return 0;
}

