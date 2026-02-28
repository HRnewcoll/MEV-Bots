// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IFlashLoan.sol";

/**
 * @title MevHelper
 * @notice On-chain helper contract for common MEV tasks:
 *   - Multi-hop price simulation (view function, no gas cost off-chain)
 *   - Batch reserve fetching for multiple pairs
 *   - Sandwich profit estimation
 *
 * Deploy this contract once and call its view functions off-chain (eth_call)
 * to simulate trades cheaply before committing to on-chain execution.
 */
contract MevHelper {
    // -------------------------------------------------------------------------
    // Interfaces
    // -------------------------------------------------------------------------

    interface IUniswapV2Pair {
        function getReserves()
            external
            view
            returns (
                uint112 reserve0,
                uint112 reserve1,
                uint32 blockTimestampLast
            );

        function token0() external view returns (address);
        function token1() external view returns (address);
    }

    interface IUniswapV2Factory {
        function getPair(address tokenA, address tokenB)
            external
            view
            returns (address pair);
    }

    // -------------------------------------------------------------------------
    // Price simulation
    // -------------------------------------------------------------------------

    /// @notice Simulate a multi-hop swap using Uniswap V2 constant-product math.
    /// @param factories  DEX factory addresses (one per hop).
    /// @param path       Token addresses (length = factories.length + 1).
    /// @param amountIn   Input amount (in path[0] decimals).
    /// @return amountOut Simulated output amount (in path[last] decimals).
    function simulateMultiHop(
        address[] calldata factories,
        address[] calldata path,
        uint256 amountIn
    ) external view returns (uint256 amountOut) {
        require(path.length == factories.length + 1, "path/factories mismatch");
        amountOut = amountIn;
        for (uint256 i = 0; i < factories.length; i++) {
            address pair = IUniswapV2Factory(factories[i]).getPair(path[i], path[i + 1]);
            require(pair != address(0), "pair not found");

            (uint112 r0, uint112 r1, ) = IUniswapV2Pair(pair).getReserves();
            address t0 = IUniswapV2Pair(pair).token0();

            (uint256 rIn, uint256 rOut) = (path[i] == t0)
                ? (uint256(r0), uint256(r1))
                : (uint256(r1), uint256(r0));

            amountOut = _getAmountOut(amountOut, rIn, rOut);
        }
    }

    /// @notice Batch fetch reserves for multiple pairs in a single call.
    /// @param pairs Array of Uniswap V2 pair addresses.
    /// @return reserve0s  Array of reserve0 values.
    /// @return reserve1s  Array of reserve1 values.
    function batchGetReserves(address[] calldata pairs)
        external
        view
        returns (uint112[] memory reserve0s, uint112[] memory reserve1s)
    {
        reserve0s = new uint112[](pairs.length);
        reserve1s = new uint112[](pairs.length);
        for (uint256 i = 0; i < pairs.length; i++) {
            (reserve0s[i], reserve1s[i], ) = IUniswapV2Pair(pairs[i]).getReserves();
        }
    }

    /// @notice Estimate sandwich profit given victim swap parameters.
    /// @param pairAddress  The AMM pair the victim is trading on.
    /// @param tokenIn      Token the victim is selling.
    /// @param victimAmountIn  Victim's input amount.
    /// @param ourAmountIn     Our front-run input amount.
    /// @return frontRunOut    Token received from our front-run.
    /// @return backRunOut     Token received from our back-run.
    /// @return grossProfit    Gross profit (backRunOut - ourAmountIn).
    function estimateSandwich(
        address pairAddress,
        address tokenIn,
        uint256 victimAmountIn,
        uint256 ourAmountIn
    )
        external
        view
        returns (
            uint256 frontRunOut,
            uint256 backRunOut,
            int256 grossProfit
        )
    {
        (uint112 r0, uint112 r1, ) = IUniswapV2Pair(pairAddress).getReserves();
        address t0 = IUniswapV2Pair(pairAddress).token0();

        (uint256 rIn, uint256 rOut) = (tokenIn == t0)
            ? (uint256(r0), uint256(r1))
            : (uint256(r1), uint256(r0));

        // After our front-run
        frontRunOut = _getAmountOut(ourAmountIn, rIn, rOut);
        uint256 rIn1 = rIn + ourAmountIn;
        uint256 rOut1 = rOut - frontRunOut;

        // After victim swap
        uint256 victimOut = _getAmountOut(victimAmountIn, rIn1, rOut1);
        uint256 rIn2 = rIn1 + victimAmountIn;
        uint256 rOut2 = rOut1 - victimOut;

        // Our back-run (sell frontRunOut back)
        backRunOut = _getAmountOut(frontRunOut, rOut2, rIn2);

        grossProfit = int256(backRunOut) - int256(ourAmountIn);
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    function _getAmountOut(
        uint256 amountIn,
        uint256 reserveIn,
        uint256 reserveOut
    ) internal pure returns (uint256) {
        require(amountIn > 0, "amountIn = 0");
        require(reserveIn > 0 && reserveOut > 0, "empty reserves");
        uint256 amountInWithFee = amountIn * 997;
        uint256 numerator = amountInWithFee * reserveOut;
        uint256 denominator = reserveIn * 1000 + amountInWithFee;
        return numerator / denominator;
    }
}
