// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./interfaces/IFlashLoan.sol";

/**
 * @title FlashLoanArbitrage
 * @notice Performs atomic flash loan–powered arbitrage across two Uniswap V2–style DEXs.
 *
 * Flow
 * ----
 * 1. Off-chain bot calls `executeArbitrage()`.
 * 2. Contract borrows `amount` of `asset` from Aave V3 (0 % upfront capital needed).
 * 3. In `executeOperation()` callback:
 *    a. Swap `asset` → `tokenOut` on dexA (cheaper).
 *    b. Swap `tokenOut` → `asset` on dexB (more expensive).
 *    c. Repay `amount + premium` to Aave.
 *    d. Transfer profit to `owner`.
 * 4. If step (c) would fail (profit < fee), the entire transaction reverts atomically.
 *
 * @dev Deploy once; call `executeArbitrage()` whenever an opportunity is detected.
 */
contract FlashLoanArbitrage is IFlashLoanReceiver {
    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    /// @notice Contract owner — receives all profits.
    address public immutable owner;

    /// @notice Aave V3 Pool (chain-specific, set at deploy time).
    IPool public immutable aavePool;

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error NotOwner();
    error NotAavePool();
    error InsufficientProfit(uint256 repayAmount, uint256 received);

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event ArbitrageExecuted(
        address indexed asset,
        uint256 flashLoanAmount,
        uint256 profit,
        address dexA,
        address dexB
    );

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    /**
     * @param _aavePool Aave V3 Pool address for the deployment chain.
     *
     * Ethereum mainnet:  0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2
     * Polygon mainnet:   0x794a61358D6845594F94dc1DB02A252b5b4814aD
     * Arbitrum mainnet:  0x794a61358D6845594F94dc1DB02A252b5b4814aD
     */
    constructor(address _aavePool) {
        owner = msg.sender;
        aavePool = IPool(_aavePool);
    }

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    /**
     * @notice Trigger a flash loan arbitrage.
     * @param asset     The token to borrow and use as base currency.
     * @param amount    How much to borrow (in `asset` decimals).
     * @param dexA      Router of the DEX to buy `tokenOut` on (should be cheaper).
     * @param dexB      Router of the DEX to sell `tokenOut` on (should be more expensive).
     * @param tokenIn   The borrowed asset address (same as `asset`).
     * @param tokenOut  The intermediate token to swap through.
     */
    function executeArbitrage(
        address asset,
        uint256 amount,
        address dexA,
        address dexB,
        address tokenIn,
        address tokenOut
    ) external {
        if (msg.sender != owner) revert NotOwner();

        address[] memory assets = new address[](1);
        assets[0] = asset;

        uint256[] memory amounts = new uint256[](1);
        amounts[0] = amount;

        // 0 = no debt (standard flash loan; repay in same tx)
        uint256[] memory modes = new uint256[](1);
        modes[0] = 0;

        // Encode strategy params to pass through the callback
        bytes memory params = abi.encode(dexA, dexB, tokenIn, tokenOut);

        aavePool.flashLoan(
            address(this),
            assets,
            amounts,
            modes,
            address(this),
            params,
            0 // referral code
        );
    }

    // -------------------------------------------------------------------------
    // Aave callback
    // -------------------------------------------------------------------------

    /**
     * @notice Called by Aave V3 after the flash loan funds are sent to this contract.
     * @dev Must repay `amounts[0] + premiums[0]` of `assets[0]` before returning.
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external override returns (bool) {
        if (msg.sender != address(aavePool)) revert NotAavePool();
        if (initiator != address(this)) revert NotAavePool();

        (address dexA, address dexB, address tokenIn, address tokenOut) =
            abi.decode(params, (address, address, address, address));

        uint256 borrowed = amounts[0];
        uint256 repayAmount = borrowed + premiums[0];

        // --- Leg 1: tokenIn → tokenOut on dexA ---
        IERC20(tokenIn).approve(dexA, borrowed);
        address[] memory pathA = new address[](2);
        pathA[0] = tokenIn;
        pathA[1] = tokenOut;
        uint256[] memory amountsA = IUniswapV2Router(dexA).swapExactTokensForTokens(
            borrowed,
            1, // accept any amount (profit check below)
            pathA,
            address(this),
            block.timestamp + 60
        );
        uint256 tokenOutReceived = amountsA[amountsA.length - 1];

        // --- Leg 2: tokenOut → tokenIn on dexB ---
        IERC20(tokenOut).approve(dexB, tokenOutReceived);
        address[] memory pathB = new address[](2);
        pathB[0] = tokenOut;
        pathB[1] = tokenIn;
        uint256[] memory amountsB = IUniswapV2Router(dexB).swapExactTokensForTokens(
            tokenOutReceived,
            repayAmount, // must receive at least enough to repay the flash loan
            pathB,
            address(this),
            block.timestamp + 60
        );
        uint256 tokenInReceived = amountsB[amountsB.length - 1];

        if (tokenInReceived < repayAmount) {
            revert InsufficientProfit(repayAmount, tokenInReceived);
        }

        uint256 profit = tokenInReceived - repayAmount;

        // --- Repay Aave ---
        IERC20(tokenIn).approve(address(aavePool), repayAmount);

        // --- Send profit to owner ---
        if (profit > 0) {
            IERC20(tokenIn).transfer(owner, profit);
        }

        emit ArbitrageExecuted(tokenIn, borrowed, profit, dexA, dexB);
        return true;
    }

    // -------------------------------------------------------------------------
    // Admin
    // -------------------------------------------------------------------------

    /// @notice Withdraw any ERC-20 tokens accidentally sent to this contract.
    function rescueToken(address token, uint256 amount) external {
        if (msg.sender != owner) revert NotOwner();
        IERC20(token).transfer(owner, amount);
    }

    /// @notice Reject accidental ETH transfers.
    receive() external payable {
        revert("No ETH accepted");
    }
}
