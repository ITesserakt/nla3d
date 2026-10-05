{
  description = "2D/3D finite element programming framework";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.zst";
  };

  inputs.self.submodules = true;

  outputs = inputs: {
    packages = builtins.mapAttrs (
      system: pkgs:
      let
        pname = "nla3d";
        version = "0.0.0";
        src = ./.;
      in
      {
        nla3d = pkgs.stdenv.mkDerivation {
          inherit pname version src;
          nativeBuildInputs = [ pkgs.cmake ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
          ];
        };

        nla3d-with-mkl = pkgs.stdenv.mkDerivation {
          inherit pname version src;
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.mkl
          ];
          buildInputs = [ pkgs.mkl ];

          cmakeFlags = [
            "-DNLA3D_USE_MKL=ON"
          ];
        };

        default = inputs.self.packages.${system}.nla3d;
      }
    ) inputs.nixpkgs.legacyPackages;

    devShells = builtins.mapAttrs (system: pkgs: {
      default = pkgs.mkShellNoCC {
        packages = with pkgs; [
          llvmPackages.clang
          llvmPackages.openmp
          gdb
          cmake
        ];
      };
    }) inputs.nixpkgs.legacyPackages;
  };
}
